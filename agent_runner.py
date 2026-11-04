#!/usr/bin/env python
"""RCLAgent-V2 (arXiv 2605.14866, github.com/LLM4AIOps/RCLAgent-V2) on the harness.

    agents/_rca_shared/.venv/bin/python agent_runner.py   (cwd agents/rclagent_v2)

stdin : {question, system_prompt, user_prompt, compress_system_prompt, compress_user_prompt, data_dir}
stdout: one JSON line {output, trajectory, usage, rclagent_v2}; everything
        else (upstream prints, Flask logs) goes to stderr.

The upstream method runs unchanged (``upstream/``, commit in ``UPSTREAM_COMMIT``):
  * ``native_data.py`` converts the case into V2's canonical layout (cache dir = DATA_ROOT);
  * ``tool_server.py`` is imported as is and served by a real werkzeug server on a free local port
    (TOOL_SERVER_PORT) in a background thread; ``coordinator.run_all()`` runs as is (DATASET_TYPE=nezha for
    RCABench, re2ob for RCA100; MAX_AGENT_PARALLEL, RESULT_SUB_DIR=result) and talks to it over HTTP;
  * LLM calls go through V2's own ``llm.py`` (requests, LLM_API_URL = OPENAI_BASE_URL + /chat/completions,
    LLM_API_KEY, LLM_MODEL from the agent's .env).
Observation only, at call boundaries (behaviour unchanged): ``llm.requests.post`` (every LLM HTTP call ->
events.jsonl + usage tracker), ``coordinator.requests.get`` (every tool-server call -> tool_call event),
``coordinator.chat_api`` (per-trace final call skipped; failures), ``AgentPool.submit`` (copies the
caller's context into pool threads so roles/parents survive), ``DedicatedAgent`` phases and
``build_trace_graph`` / ``inspect_trace`` (event scopes).
One policy on top: an LLM call that still fails after llm.py's own 3 attempts ends the process with exit
code 3 (upstream would swallow it into an '{"error": ...}' evidence string); the harness counts it as an
infrastructure failure. Disable with RCLV2_FAIL_FAST=0.
Case-level final diagnosis: upstream's per-trace phases run unchanged for every
selected trace, but its per-trace final call (ranking via print_results) is not sent; instead ONE call per
case receives every trace's evidence blocks (Root-Level Diagnosis Report, Global Evidence Graph summary,
evidence hint, verbatim) and must call ``output_causal_graph`` (harness CausalGraph). The model's graph is
the case answer. Every intermediate result (trace tree, span verdicts, consolidations,
reports, final prompt and raw response) is saved as artifacts and in ``rclagent_v2.traces``.
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
UP = HERE / "upstream"
sys.path.insert(0, str(HERE.parent))           # agents/ -> _rca_shared
sys.path.insert(0, str(HERE))                  # native_data

_STDOUT = sys.stdout
sys.stdout = sys.stderr

import requests                                  # noqa: E402

from _rca_shared import cases                    # noqa: E402
from _rca_shared import llm as shared_llm        # noqa: E402
from _rca_shared.events import EV, install_hooks  # noqa: E402
import native_data                               # noqa: E402

FAIL_FAST = os.environ.get("RCLV2_FAIL_FAST", "1") != "0"
_TRACE: contextvars.ContextVar = contextvars.ContextVar("rclv2_trace", default=None)
_AGENT: contextvars.ContextVar = contextvars.ContextVar("rclv2_agent", default=None)
LOCK = threading.Lock()
CALLS: list = []             # one record per LLM HTTP call (for trajectory / counts)
FAILURES: list = []


def log(*a):
    print("[rclagent_v2]", *a, file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _agent_key() -> tuple:
    k = _TRACE.get()
    a = _AGENT.get()
    return (k, a[0], a[1]) if a else (k, "coordinator", "final_synthesis")


# ------------------------------------------------------------------ observation wrappers
class _LLMRequests:
    """Stands in for ``requests`` inside upstream llm.py: logs each chat-completion POST, returns it unchanged."""

    def __init__(self, tracker):
        self.tracker = tracker

    def __getattr__(self, name):
        return getattr(requests, name)

    def post(self, url, *args, **kwargs):
        payload = kwargs.get("json") or {}
        call_id = EV.new_id("llm_")
        EV.emit("llm_call_start", call_id=call_id, model=payload.get("model"))
        t0 = time.time()
        try:
            resp = requests.post(url, *args, **kwargs)
        except Exception as exc:
            EV.emit("llm_call", call_id=call_id, request=payload, response=None,
                    error=f"{type(exc).__name__}: {exc}", latency_s=time.time() - t0)
            raise
        if kwargs.get("stream"):
            EV.emit("llm_call", call_id=call_id, request=payload, response=None,
                    note="stream response not captured", latency_s=time.time() - t0)
            return resp
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            data = {"raw": resp.text, "status_code": resp.status_code}
        err = None if resp.ok else f"HTTP {resp.status_code}"
        if resp.ok:
            self.tracker.track(data)
        EV.emit("llm_call", call_id=call_id, request=payload, response=data,
                usage=data.get("usage") if isinstance(data, dict) else None, error=err,
                latency_s=time.time() - t0)
        if resp.ok:
            try:
                msg = data["choices"][0]["message"]
            except Exception:  # noqa: BLE001
                msg = {"role": "assistant", "content": str(data)}
            with LOCK:
                CALLS.append({"key": _agent_key(), "seq": len(CALLS), "messages": list(payload.get("messages") or []),
                              "tools": [t["function"]["name"] for t in payload.get("tools") or []],
                              "response": msg, "usage": data.get("usage") if isinstance(data, dict) else None})
        return resp


class _ToolRequests:
    """Stands in for ``requests`` inside coordinator.py: logs each tool-server GET, returns it unchanged."""

    def __getattr__(self, name):
        return getattr(requests, name)

    def get(self, url, *args, **kwargs):
        u = urlparse(url)
        name = u.path.strip("/")
        params = kwargs.get("params") or {k: v[0] if len(v) == 1 else v for k, v in parse_qs(u.query).items()}
        try:
            resp = requests.get(url, *args, **kwargs)
        except Exception as exc:
            EV.tool_call(name, params, result=None, error=f"{type(exc).__name__}: {exc}")
            raise
        EV.tool_call(name, params, result=resp.text, http_status=resp.status_code)
        return resp


PHASE_NAMES = {"self_state_verification": "verification", "consolidation": "consolidation"}

# ---- output-format adaptation of the final diagnosis: CausalGraph instead of a ranking
GRAPH_TOOL_NAME = "output_causal_graph"
SERVICE_STATES = ["HEALTHY", "HIGH_ERROR_RATE", "HIGH_LATENCY", "UNAVAILABLE"]   # rca.yaml, service level
_NODE_SCHEMA = {"type": "object",
                "properties": {"component": {"type": "string", "description": "Service name."},
                               "state": {"type": "array", "items": {"type": "string", "enum": SERVICE_STATES}}},
                "required": ["component", "state"]}
OUTPUT_CAUSAL_GRAPH_FUNCTION = {
    "type": "function",
    "function": {
        "name": GRAPH_TOOL_NAME,
        "description": "Output the root cause and the fault-propagation causal graph of this failure.",
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": _NODE_SCHEMA,
                          "description": "Every service in the graph with its state."},
                "edges": {"type": "array",
                          "items": {"type": "object",
                                    "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
                                    "required": ["source", "target"]},
                          "description": "Fault propagation, from cause (source) to effect (target)."},
                "root_causes": {"type": "array", "items": _NODE_SCHEMA, "minItems": 1,
                                "description": "Root-cause service(s), at least one."},
                "component_to_service": {"type": "object", "description": "Leave empty: {}."},
            },
            "required": ["nodes", "edges", "root_causes"],
        },
    },
}
EVIDENCE_START = "Root-Level Diagnosis Report (from recursive agent tree):\n"   # upstream final prompt
EVIDENCE_END = "\n\nGoal: identify the root-cause service."
GEG_START = "\n\nGlobal Evidence Graph ("
CASE_OPENER = "You are the ROOT AGENT producing the FINAL diagnosis."       # upstream final prompt, line 1
GOAL_LINE = "Goal: identify the root-cause service."                      # upstream final prompt
GRAPH_INSTRUCTIONS = """Graph instructions:
1. Name every component at service level, using the known services listed in the system prompt.
2. root_causes: the root-cause service(s) of this failure (at least one).
3. edges: the fault propagation from the root cause to the affected upstream services seen in these traces,
   each edge from cause (source) to effect (target).
4. nodes: every service in root_causes and edges, with its state.
5. state values: HEALTHY, HIGH_ERROR_RATE, HIGH_LATENCY, UNAVAILABLE.
6. You MUST call the `output_causal_graph` function.
   Do not output anything else."""
SKIPPED_FINAL = json.dumps({"root_causes": [], "note": "per-trace final call not run by the harness port; "
                                                       "one case-level diagnosis call uses this trace's evidence"})


def split_evidence(prompt: str) -> dict:
    """The evidence blocks of upstream's per-trace final prompt, verbatim."""
    i = prompt.index(EVIDENCE_START)
    j = prompt.index(EVIDENCE_END, i)
    block = prompt[i:j]
    g = block.index(GEG_START)
    return {"block": block, "root_level_report": block[len(EVIDENCE_START):g]}


# Context guard: upstream's Global Evidence Graph summary lists every span's verdict, and the
# case prompt carries it for all 5 traces; very large traces (p90 ≈ 1,800 spans) could exceed the model
# context. Over the budget, the largest traces' summaries are replaced — largest first, only as far as
# needed — by the span verdicts flagged abnormal / with error logs (the predicate upstream's final step
# uses for evidence_hint) plus a count of the omitted span verdicts. Under the budget the prompt is unchanged.
FINAL_PROMPT_CHAR_BUDGET = int(os.environ.get("RCLV2_FINAL_PROMPT_CHAR_BUDGET", "600000"))


def _abnormal_only(geg_summary: str) -> str | None:
    """Keep the span verdicts upstream's final step treats as evidence when it builds evidence_hint
    (coordinator.inspect_trace: '"has_error_logs": true' or '"is_abnormal": true'). Upstream's own
    GlobalEvidenceGraph.abnormal_nodes() also matches any text containing "abnormal", i.e. every verdict
    (they all carry an is_abnormal key), so it cannot shorten the summary."""
    try:
        nodes = json.loads(geg_summary)
    except (TypeError, json.JSONDecodeError):
        return None

    def flagged(ev) -> bool:
        t = re.sub(r"\s+", " ", ev.lower()) if isinstance(ev, str) else ""
        return any(k in t for k in ('"has_error_logs": true', '"is_abnormal": true',
                                     '"has_error_logs":true', '"is_abnormal":true'))

    keep = [n for n in nodes if flagged(n.get("evidence"))]
    return (json.dumps(keep, ensure_ascii=False, indent=2)
            + f"\n({len(nodes) - len(keep)} non-abnormal span evidences omitted to fit the model context)")


def _fit_budget(traces: list[dict], fixed: int) -> list[dict]:
    total = fixed + sum(len(t["block"]) for t in traces)
    out = [dict(t) for t in traces]
    for t in sorted(out, key=lambda x: -len(x["block"])):
        if total <= FINAL_PROMPT_CHAR_BUDGET:
            break
        g = t.get("geg_summary")
        compact = _abnormal_only(g) if g else None
        if g and compact is not None and g in t["block"]:
            new = t["block"].replace(g, compact, 1)
            total -= len(t["block"]) - len(new)
            t["block"], t["geg_compacted"] = new, True
    return out


def case_final_prompt(traces: list[dict]) -> str:
    """One case-level final diagnosis prompt: upstream's opener, then every trace's evidence blocks under a
    short header, then the goal line and the graph instructions."""
    traces = _fit_budget(traces, len(CASE_OPENER) + len(GOAL_LINE) + len(GRAPH_INSTRUCTIONS))
    compacted = [t["k"] for t in traces if t.get("geg_compacted")]
    if compacted:
        EV.artifact("final_prompt_compacted", {"traces": compacted, "budget_chars": FINAL_PROMPT_CHAR_BUDGET})
    parts = [CASE_OPENER, f"The failure was analysed along {len(traces)} error trace(s)."]
    for t in traces:
        parts.append(f"=== Trace {t['k'] + 1}/{len(traces)} (root operation: {t['operation']}) ===\n{t['block']}")
    parts += [GOAL_LINE, GRAPH_INSTRUCTIONS]
    return "\n\n".join(parts)


def _clean_graph(g):
    """Harness CausalGraph shape; None if the reply has no usable root cause."""
    if not isinstance(g, dict):
        return None

    def node(x):
        if isinstance(x, str):
            return {"component": x, "state": []}
        if isinstance(x, dict) and str(x.get("component") or "").strip():
            st = x.get("state") or []
            return {"component": str(x["component"]).strip(), "state": [str(v) for v in (st if isinstance(st, list) else [st])]}
        return None

    roots = [n for n in map(node, g.get("root_causes") or []) if n]
    if not roots:
        return None
    nodes = [n for n in map(node, g.get("nodes") or []) if n]
    edges = [{"source": str(e["source"]).strip(), "target": str(e["target"]).strip()}
             for e in g.get("edges") or [] if isinstance(e, dict) and e.get("source") and e.get("target")]
    c2s = g.get("component_to_service") if isinstance(g.get("component_to_service"), dict) else {}
    return {"nodes": nodes, "edges": edges, "root_causes": roots, "component_to_service": c2s}
TRACES: dict = {}            # k -> intermediate results of that error trace (tree, verdicts, consolidations, ...)


def _trace_store(k) -> dict:
    with LOCK:
        return TRACES.setdefault(k, {"tree": [], "verdicts": {}, "consolidations": {},
                                     "final_prompt": None, "geg_summary": None,
                                     "evidence_hint": None, "result": None})


def _parse_json(text: str):
    """V2's own JSON-from-text idea (first '{' .. last '}'); None if it does not parse."""
    try:
        a, b = text.find("{"), text.rfind("}") + 1
        return json.loads(text[a:b]) if a >= 0 and b > a else None
    except Exception:  # noqa: BLE001
        return None


def _graph_args(args):
    """The graph tool's arguments as a dict. A reply may nest the whole payload in the `nodes` string
    ('[...], "edges": [...], "root_causes": [...]'); that string is the rest of the object, re-read as one."""
    g = _parse_json(args) if isinstance(args, str) else args
    if isinstance(g, dict) and not g.get("root_causes") and isinstance(g.get("nodes"), str):
        nested = _parse_json('{"nodes": ' + g["nodes"].strip() + "}")
        if isinstance(nested, dict) and nested.get("root_causes"):
            return nested
    return g


def _flatten_tree(node: dict, parent=None, depth: int = 0, out=None) -> list:
    out = [] if out is None else out
    raw = node.get("raw", {}) or {}
    out.append({"span_id": node.get("span_id"), "parent": parent, "depth": depth,
                "cmdb_id": raw.get("cmdb_id"), "operation_name": raw.get("operation_name"),
                "duration": raw.get("duration"), "status_code": raw.get("status_code"),
                "timestamp": raw.get("timestamp")})
    for c in node.get("children", []):
        _flatten_tree(c, node.get("span_id"), depth + 1, out)
    return out


def _install(coord, llm_mod, tracker, span2k: dict):
    llm_mod.requests = _LLMRequests(tracker)
    coord.requests = _ToolRequests()

    orig_chat = coord.chat_api

    def call(prompts, tools):
        # every LLM call's role names trace, span and phase (verification / consolidation / fallback / final)
        k, agent = _TRACE.get(), _AGENT.get()
        if agent:
            ph = PHASE_NAMES[agent[1]]
            if ph == "verification" and not tools:
                ph = "fallback"
            role = f"rclv2:trace-{k}:span-{agent[0]}:{ph}"
        elif k == "case":
            role = "rclv2:case:final"
        else:
            role = f"rclv2:trace-{k}:final"
        try:
            with EV.scope(role=role):
                return orig_chat(prompts, tools)
        except Exception as exc:
            FAILURES.append(f"{type(exc).__name__}: {exc}")
            log(f"LLM call failed after llm.py retries: {type(exc).__name__}: {exc}")
            if FAIL_FAST:
                traceback.print_exc()
                sys.stderr.flush()
                os._exit(3)                         # infrastructure failure: no result line
            raise

    def chat_api(prompts, tools):
        agent = _AGENT.get()
        phase = agent[1] if agent else "final_synthesis"
        k = _TRACE.get()
        if phase == "final_synthesis" and any(t["function"]["name"] == "print_results" for t in tools or []):
            # case-level diagnosis: the per-trace final call is not run; its evidence is kept
            st = _trace_store(k)
            st["final_prompt"] = prompts[1]["content"]
            st.update(split_evidence(st["final_prompt"]))
            return SKIPPED_FINAL, []                 # upstream's text fallback writes this to the result file
        return call(prompts, tools)

    coord.chat_api = chat_api

    orig_submit = coord.AgentPool.submit

    def submit(self, fn, *args, **kwargs):          # contextvars do not cross into pool threads
        ctx = contextvars.copy_context()
        return orig_submit(self, ctx.run, fn, *args, **kwargs)

    coord.AgentPool.submit = submit

    def phase(method, name):
        store_key = {"self_state_verification": "verdicts", "consolidation": "consolidations"}[name]

        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            k = _TRACE.get()
            tok = _AGENT.set((self.span_id, name))
            try:
                with EV.scope(role=f"rclv2:trace-{k}:span-{self.span_id}:{PHASE_NAMES[name]}",
                              parent=f"trace-{k}/span-{self.span_id}"):
                    out = method(self, *args, **kwargs)
                raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
                _trace_store(k)[store_key][self.span_id] = {"raw": raw, "parsed": _parse_json(raw)}
                return out
            finally:
                _AGENT.reset(tok)
        return wrapped

    coord.DedicatedAgent.self_state_verification = phase(coord.DedicatedAgent.self_state_verification,
                                                         "self_state_verification")
    coord.DedicatedAgent.consolidate = phase(coord.DedicatedAgent.consolidate, "consolidation")

    orig_geg = coord.GlobalEvidenceGraph

    class RecordedEvidenceGraph(orig_geg):          # same class, instance kept for the artifacts
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            _trace_store(_TRACE.get())["geg_obj"] = self

    coord.GlobalEvidenceGraph = RecordedEvidenceGraph

    orig_build, orig_inspect = coord.build_trace_graph, coord.inspect_trace

    def build_trace_graph(root_span_id, *a, **kw):
        k = span2k.get(str(root_span_id).strip())
        tok = _TRACE.set(k)
        try:
            with EV.scope(role=f"rclv2:trace-{k}:trace_graph", parent=f"trace-{k}"):
                graph = orig_build(root_span_id, *a, **kw)
            _trace_store(k)["tree"] = _flatten_tree(graph)
            return graph
        finally:
            _TRACE.reset(tok)

    def inspect_trace(trace_graph, *a, **kw):
        k = span2k.get(str(trace_graph.get("span_id", "")).strip())
        tok = _TRACE.set(k)
        try:
            with EV.scope(role=f"rclv2:trace-{k}:final", parent=f"trace-{k}"):
                res = orig_inspect(trace_graph, *a, **kw)
            st = _trace_store(k)
            st["result"] = res
            geg = st.pop("geg_obj", None)
            if geg is not None:
                st["geg_summary"] = geg.summary()
                fp = st["final_prompt"] or ""
                if st["geg_summary"] in fp:          # the hint exactly as the coordinator rendered it
                    tail = fp.split(st["geg_summary"], 1)[1].split("\n\nGoal:", 1)[0]
                    st["evidence_hint"] = tail[1:] if tail.startswith("\n") else tail
            EV.artifact("trace_result_file", {"k": k, "root_span": trace_graph.get("span_id"), "result": res})
            return res
        finally:
            _TRACE.reset(tok)

    coord.build_trace_graph = build_trace_graph
    coord.inspect_trace = inspect_trace
    return call


# ------------------------------------------------------------------ metric tool windows
def install_metric_windows(tool_server, abnormal_start_s: int) -> None:
    """``search_fluctuating_metrics`` with the case's two windows (RCABench and RCA100).

    Upstream takes the "regular" statistics over ts ± 1200 s and the "current" points over ts ± 600 s of one
    metric table: on its datasets (hours to days of 1-minute data around a short fault) the regular range
    is mostly normal time. An RCABench case spans about 8 minutes, so both ranges hold the whole case and
    regular_mean always equals current_mean. Here the regular range is the ts ± 1200 s rows before the
    abnormal window starts (normal window) and the current range the ts ± 600 s rows from its start on
    (abnormal window). Service matching, the 3-sigma spike test, keys and the CSV output are upstream's,
    line for line."""
    import pandas as pd
    from flask import jsonify, request

    metric_df = tool_server.metric_df
    service_node_map = tool_server.service_node_map
    normal = metric_df["timestamp"] < abnormal_start_s

    def search_fluctuating_metrics():
        service_name  = request.args.get("service_name")
        timestamp_str = request.args.get("timestamp")
        if not service_name:
            return jsonify({"error": "service_name parameter missing"}), 400
        if not timestamp_str:
            return jsonify({"error": "timestamp parameter missing"}), 400
        try:
            ts = int(str(timestamp_str)[:10])
        except ValueError:
            return jsonify({"error": "Invalid timestamp format"}), 400

        node_id = service_node_map.get(service_name)
        if node_id:
            cond = (
                metric_df["service_name"].str.contains(service_name, case=False, na=False) |
                (
                    (metric_df["node_id"] == node_id) &
                    (metric_df["service_name"].fillna("") == "")
                )
            )
        else:
            cond = metric_df["service_name"].str.contains(service_name, case=False, na=False)

        matched = metric_df[cond]
        if matched.empty:
            return jsonify({"message": "No matching records found."}), 200
        baseline_df = matched[normal[cond] & matched["timestamp"].between(ts - 1200, ts + 1200)]
        current_df = matched[~normal[cond] & matched["timestamp"].between(ts - 600, ts + 600)]

        kpi_dict = {}
        for (kpi, nid, svc), group in baseline_df.groupby(["kpi_name", "node_id", "service_name"]):
            mean_val = group["value"].mean()
            std_val  = group["value"].std()
            window   = current_df[(current_df["kpi_name"] == kpi) & (current_df["node_id"] == nid)
                                  & (current_df["service_name"] == svc)]
            if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                continue
            threshold = 3 * std_val
            is_spike  = (
                (window["value"] < mean_val - threshold) |
                (window["value"] > mean_val + threshold)
            ).any()
            if not is_spike:
                continue
            key = kpi
            if not pd.isna(svc) and svc:
                key = f"{svc}.{key}"
            if not pd.isna(nid) and nid:
                key = f"{nid}.{key}"
            kpi_dict[key] = {
                "regular_mean":    round(mean_val, 2),
                "regular_std_dev": round(std_val, 2),
                "current_mean":    round(window["value"].mean(), 2),
                "current_std_dev": round(window["value"].std(), 2),
            }

        if not kpi_dict:
            return jsonify({"message": "No fluctuating metrics found."}), 200

        rows   = [["key", "regular_mean", "regular_std_dev", "current_mean", "current_std_dev"]]
        rows  += [[k] + list(v.values()) for k, v in kpi_dict.items()]
        df_out = pd.DataFrame(rows[1:], columns=rows[0])
        return df_out.to_csv(index=False)

    tool_server.app.view_functions["search_fluctuating_metrics"] = search_fluctuating_metrics


# ------------------------------------------------------------------ log tool result guard (config)
# Upstream's search_logs returns every non-INFO/DEBUG line of the matching pods within ±60 s, unbounded. On
# RCA100 the frontend's JavaScript error storms (a dozen stack lines per error, no level) reach several MB per
# call. RCLV2_LOG_RESULT_CHAR_BUDGET > 0 bounds a result only when it is larger: lines of one pod with the same
# template are merged, then rows are cut. 0 (default) = upstream's result unchanged.
LOG_RESULT_CHAR_BUDGET = int(os.environ.get("RCLV2_LOG_RESULT_CHAR_BUDGET", "0"))
LOG_GUARD: list = []         # one record per bounded result
_SEVERITY_RANK = {"fatal": 0, "critical": 0, "severe": 0, "error": 0, "warn": 1, "warning": 1}


def log_template(value: str) -> str:
    """Line with its variable parts masked: UUIDs and hex ids (>= 6 hex chars with a digit) -> <ID>, digit
    runs -> <N>, whitespace squashed."""
    s = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b", "<ID>", value)
    s = re.sub(r"(?<![0-9A-Za-z])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}(?![0-9A-Za-z])", "<ID>", s)
    s = re.sub(r"\d+", "<N>", s)
    return re.sub(r"\s+", " ", s).strip()


def bound_log_result(text: str, budget: int) -> tuple[str, dict | None]:
    """``search_logs`` CSV larger than ``budget`` characters.

    1. Merge: rows of one pod (cmdb_id, log_name) whose values share a template (``log_template``) are
       merged into their first occurrence, verbatim, with ``repeat_count`` = number of merged rows.
    2. Cut, if the merged CSV is still larger: rows are kept by severity (error-like, then warn, then the
       rest) and within a severity by repeat_count, as many as fit; the kept rows stay in time order and a
       last line gives the number of omitted rows. A first line that alone exceeds the budget is kept with
       its text cut.
    Results within the budget are returned unchanged."""
    import io

    import pandas as pd

    if budget <= 0 or len(text) <= budget:
        return text, None
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    key = df["cmdb_id"] + "\x1f" + df["log_name"] + "\x1f" + df["value"].map(log_template)
    first = ~key.duplicated()
    out = df[first].assign(repeat_count=key[first].map(key.value_counts()).values)
    merged = out.to_csv(index=False)
    kept = len(out)
    if len(merged) > budget:
        sev = out["value"].str.extract(r"^severity:\s*(\w+)", expand=False).str.lower()
        rank = sev.map(_SEVERITY_RANK).fillna(2)
        size = out.apply(lambda r: len(",".join(map(str, r))) + 3, axis=1)
        order = pd.DataFrame({"rank": rank, "count": -out["repeat_count"], "pos": range(len(out))},
                             index=out.index).sort_values(["rank", "count", "pos"]).index
        room = budget - len(merged.splitlines()[0]) - 120
        take = []
        for ix in order:
            if size[ix] > room:
                break
            take.append(ix)
            room -= size[ix]
        if not take and len(order):                       # the first line alone is too long: cut its text
            ix = order[0]
            v = out.at[ix, "value"]
            out.at[ix, "value"] = v[:max(room - (size[ix] - len(v)) - 1, 100)] + "…"
            take = [ix]
        rest = out.drop(index=take)
        kept = len(take)
        merged = (out.loc[sorted(take, key=out.index.get_loc)].to_csv(index=False)
                  + f"# {len(rest)} more distinct lines ({int(rest['repeat_count'].sum())} rows) omitted "
                    f"to fit {budget} characters\n")
    return merged, {"rows": len(df), "distinct_rows": len(out), "kept_rows": kept,
                    "chars_before": len(text), "chars_after": len(merged)}


def install_log_result_guard(tool_server, budget: int) -> None:
    """Wrap upstream's search_logs view (tool_server.py unedited) with ``bound_log_result``."""
    orig = tool_server.app.view_functions["search_logs"]

    def search_logs():
        resp = orig()
        if not isinstance(resp, str):                    # upstream's JSON error responses
            return resp
        text, rec = bound_log_result(resp, budget)
        if rec is not None:
            from flask import request
            rec.update(service_name=request.args.get("service_name"), timestamp=request.args.get("timestamp"))
            with LOCK:
                LOG_GUARD.append(rec)
        return text

    tool_server.app.view_functions["search_logs"] = search_logs


# ------------------------------------------------------------------ system prompt service block
# Upstream dataset type per benchmark: RCABench is TrainTicket (upstream "nezha"); RCA100 is the
# OpenTelemetry Demo, a fork of Online Boutique (upstream "re2ob" = RCAEval RE2-OB, Online Boutique).
DATASET_TYPES = {"rcabench": "nezha", "rca100": "re2ob"}
SERVICE_BLOCKS = {        # (block start, block end) of each type's system prompt; only the block between them changes
    "nezha": ("The system is a TrainTicket microservice application. Known services:\n", "\nPods have K8s names like"),
    "re2ob": ("The system is an Online Boutique (HipsterShop) microservice application. Known services:\n",
              "\nUse the exact service names as they appear in trace data."),
}
# RCA100 is not Online Boutique: the system name in the re2ob prompt is replaced (data-bound, like the list)
SYSTEM_NAMES = {"re2ob": ("an Online Boutique (HipsterShop) microservice application",
                          "an OpenTelemetry Demo (Astronomy Shop) microservice application")}
SERVICE_BLOCK_START, SERVICE_BLOCK_END = SERVICE_BLOCKS["nezha"]
SERVICE_BLOCK_WIDTH = 80            # longest line of upstream's (hand-wrapped) nezha block, incl. indent and comma


def format_service_block(services: list[str], width: int = SERVICE_BLOCK_WIDTH) -> str:
    """Upstream's layout: two-space indent, ', '-separated, lines wrapped at `width`, no final comma."""
    lines, cur = [], []
    for name in services:
        cand = "  " + ", ".join(cur + [name]) + ","
        if cur and len(cand) > width:
            lines.append("  " + ", ".join(cur) + ",")
            cur = [name]
        else:
            cur.append(name)
    if cur:
        lines.append("  " + ", ".join(cur))
    return "\n".join(lines)


def patch_service_block(prompt: str, services: list[str], dataset_type: str = "nezha") -> str:
    """Replace only the service-name block of coordinator's system prompt for ``dataset_type`` (data-bound
    list), in upstream's layout (lines as long as the block's longest upstream line); for re2ob also the
    system name."""
    start, end = SERVICE_BLOCKS[dataset_type]
    i = prompt.index(start) + len(start)
    j = prompt.index(end, i)
    width = max(len(line) for line in prompt[i:j].split("\n"))
    out = prompt[:i] + format_service_block(services, width) + prompt[j:]
    if dataset_type in SYSTEM_NAMES:
        old, new = SYSTEM_NAMES[dataset_type]
        assert out.count(old) == 1, f"upstream {dataset_type} prompt changed: {old!r}"
        out = out.replace(old, new)
    return out


# ------------------------------------------------------------------ answer (GT-free) + causal graph
def _to_service(candidate: str, instances: list[str]) -> str:
    """Candidate (service / pod / node) -> RCABench service, '' if it names none (e.g. a node)."""
    v = str(candidate or "").strip().lower()
    if not v:
        return ""
    low = {i.lower(): i for i in instances}
    if v in low:
        return low[v]
    for i in sorted(instances, key=len, reverse=True):          # pod name -> its service
        if v.startswith(i.lower() + "-"):
            return i
    return ""


def _canon(component: str, instances: list[str]) -> str:
    """Component name -> RCABench service when it names one (service or pod), else the name as given."""
    return _to_service(component, instances) or str(component).strip()


def canonical_graph(g: dict | None, instances: list[str]) -> dict:
    """The model's graph as the case answer: component names canonicalised to services (pod -> service),
    duplicate nodes/edges and self-loops dropped; nothing added."""
    if not g:
        return _graph([], [])
    states: dict = {}
    order: list = []

    def add(comp, st):
        c = _canon(comp, instances)
        if c not in order:
            order.append(c)
        cur = states.setdefault(c, [])
        cur.extend(v for v in st or [] if v not in cur)
        return c

    roots = []
    for r in g["root_causes"]:
        c = add(r["component"], r["state"])
        if c not in roots:
            roots.append(c)
    for n in g["nodes"]:
        add(n["component"], n["state"])
    edges = []
    for e in g["edges"]:
        a, b = add(e["source"], []), add(e["target"], [])
        if a != b and [a, b] not in edges:
            edges.append([a, b])
    node = lambda c: {"component": c, "state": states.get(c) or ["UNKNOWN"]}  # noqa: E731
    return {"nodes": [node(c) for c in order], "edges": [{"source": a, "target": b} for a, b in edges],
            "root_causes": [node(c) for c in roots], "component_to_service": {}}


def _graph(roots: list[str], edges: list) -> dict:
    services = list(dict.fromkeys(list(roots) + [s for e in edges for s in e]))
    return {"nodes": [{"component": s, "state": ["UNKNOWN"]} for s in services],
            "edges": [{"source": a, "target": b} for a, b in edges],
            "root_causes": [{"component": s, "state": ["UNKNOWN"]} for s in roots],
            "component_to_service": {}}


def _trajectory() -> tuple[list, dict]:
    """Per agent (trace, span, phase): the last request's messages + the final reply, flattened in call order."""
    last, first, counts = {}, {}, Counter()
    for c in CALLS:
        last[c["key"]] = c
        first.setdefault(c["key"], c["seq"])
        counts[c["key"][0]] += 1
    out = []
    for key in sorted(last, key=lambda x: first[x]):
        k, span, ph = key
        label = ("rclv2:case:final" if k == "case" else
                 f"rclv2:trace-{k}:{'coordinator' if span == 'coordinator' else 'span-' + span}:{ph}")
        for m in last[key]["messages"] + [last[key]["response"]]:
            out.append({**m, "agent": label})
    return out, dict(counts)


# ------------------------------------------------------------------ case
def run_case(data_dir: Path, tracker) -> dict:
    dataset = cases.kind(data_dir)
    dtype = DATASET_TYPES[dataset]
    instances = cases.instances_of(data_dir)
    native = native_data.convert(data_dir)
    log(native_data.summary(native))
    EV.artifact("rclv2_native_data", {k: v for k, v in native.items() if k != "error_traces"})
    EV.artifact("error_traces", native["error_traces"])
    root = Path(native["root"])
    lines = (root / "error_traces.txt").read_text().splitlines(keepends=True)
    span2k, entries = {}, []
    for i, line in enumerate(lines[1:], start=1):                   # coordinator.run_all's parsing
        parts = line.split()
        if len(parts) < 4 or not parts[3].strip():
            continue
        span2k[parts[3].strip()] = i - 1
        entries.append({"k": i - 1, "index": i, "root_span": parts[3].strip(), "trace_id": parts[4],
                        "timestamp": parts[1]})
    meta = {"native_root": str(root), "error_traces": {k: v for k, v in native["error_traces"].items() if k != "rows"},
            "n_traces": len(entries), "dataset": dataset, "dataset_type": dtype, "known_services": list(instances)}
    if not entries:
        log("no error traces; finishing without an answer")
        meta["note"] = "no error traces"
        return {"graph": _graph([], []), "trajectory": [], "meta": meta}

    result_sub = "result"
    shutil.rmtree(root / result_sub, ignore_errors=True)            # run_all skips traces with a result file
    s = shared_llm.settings()
    port = _free_port()
    os.environ.update({
        "DATA_ROOT": str(root), "DATASET_TYPE": dtype, "RESULT_SUB_DIR": result_sub,
        "MAX_AGENT_PARALLEL": os.environ.get("RCLV2_MAX_AGENT_PARALLEL", "8"),
        "TOOL_SERVER_HOST": "127.0.0.1", "TOOL_SERVER_PORT": str(port),
        "LLM_API_URL": s["base_url"].rstrip("/") + "/chat/completions",
        "LLM_API_KEY": s["api_key"], "LLM_MODEL": s["model"],
    })
    sys.path.insert(0, str(UP))
    import config                                   # noqa: E402  (reads the env above)
    # The case's service list for its DATASET_TYPE (runtime only; upstream/config.py is not edited).
    # coordinator binds _KNOWN_SERVICES = config.get_known_services() at import, so patch first.
    config.KNOWN_SERVICES[dtype] = list(instances)
    import llm as up_llm                            # noqa: E402
    import tool_server                              # noqa: E402  (loads DATA_ROOT)
    abnormal_start_s = int(native["error_traces"]["window_ms"][0]) // 1000
    install_metric_windows(tool_server, abnormal_start_s)
    EV.artifact("metric_windows", {"abnormal_start_s": abnormal_start_s,
                                   "regular": "ts ± 1200 s, before abnormal_start_s",
                                   "current": "ts ± 600 s, from abnormal_start_s on"})
    if LOG_RESULT_CHAR_BUDGET > 0:
        install_log_result_guard(tool_server, LOG_RESULT_CHAR_BUDGET)
        EV.artifact("log_result_guard", {"budget_chars": LOG_RESULT_CHAR_BUDGET})
    import coordinator as coord                     # noqa: E402
    assert coord._KNOWN_SERVICES == list(instances), "KNOWN_SERVICES injection must precede the coordinator import"
    EV.artifact("known_services_injected", {"dataset_type": dtype, "services": list(instances)})
    upstream_prompt = coord.SYSTEM_PROMPT
    assert upstream_prompt == coord._SYSTEM_PROMPTS[dtype] + coord._COMMON_GUIDELINES, "DATASET_TYPE not applied"
    coord._SYSTEM_PROMPTS[dtype] = patch_service_block(coord._SYSTEM_PROMPTS[dtype], instances, dtype)
    coord.SYSTEM_PROMPT = coord._SYSTEM_PROMPTS[dtype] + coord._COMMON_GUIDELINES   # read at call time
    EV.artifact("system_prompt_patched", {"prompt": coord.SYSTEM_PROMPT, "upstream_prompt": upstream_prompt})
    from werkzeug.serving import make_server

    srv = make_server("127.0.0.1", port, tool_server.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    call = _install(coord, up_llm, tracker, span2k)
    t0 = time.time()
    try:
        coord.run_all(str(root))
    finally:
        srv.shutdown()
    results, missing, traces, evidence = [], [], [], []
    rows = native["error_traces"].get("rows") or []
    for e in entries:
        k = e["k"]
        f = root / result_sub / f"conversation_trace_{e['index']}.txt"
        if f.exists():                                           # upstream's result file (skip note)
            results.append({**e, "result": json.loads(f.read_text())})
        else:
            missing.append(k)
        st = _trace_store(k)
        operation = rows[k]["operation_name"] if k < len(rows) else ""
        if st.get("block"):
            evidence.append({"k": k, "operation": operation, "block": st["block"],
                             "geg_summary": st.get("geg_summary")})
        with EV.scope(parent=f"trace-{k}"):             # every intermediate result of this trace
            EV.artifact("trace_tree", {"k": k, "tree": st["tree"]})
            EV.artifact("span_verdicts", {"k": k, "verdicts": st["verdicts"]})
            EV.artifact("span_consolidations", {"k": k, "consolidations": st["consolidations"]})
            EV.artifact("trace_report", {"k": k, "root_level_report": st.get("root_level_report"),
                                         "geg_summary": st["geg_summary"], "evidence_hint": st["evidence_hint"],
                                         "upstream_final_prompt_not_sent": st["final_prompt"],
                                         "result_file": st["result"]})
        traces.append({
            "k": k, "root_span": e["root_span"], "trace_id": e["trace_id"], "operation": operation,
            "tree": [{x: n[x] for x in ("span_id", "parent", "depth", "cmdb_id", "operation_name")} for n in st["tree"]],
            "verdicts": {sid: v["parsed"] if v["parsed"] is not None else v["raw"] for sid, v in st["verdicts"].items()},
            "consolidations": {sid: v["parsed"] if v["parsed"] is not None else v["raw"]
                               for sid, v in st["consolidations"].items()},
            "root_level_report": st.get("root_level_report"), "evidence_hint": st["evidence_hint"]})

    # ── case-level final diagnosis: one call over every trace's evidence blocks ──
    final = {"prompt": None, "content": None, "tool_calls": None, "graph": None, "source": None}
    if evidence:
        final["prompt"] = case_final_prompt(evidence)
        messages = [{"role": "system", "content": coord.SYSTEM_PROMPT}, {"role": "user", "content": final["prompt"]}]
        tok = _TRACE.set("case")
        try:
            content, tool_calls = call(messages, [OUTPUT_CAUSAL_GRAPH_FUNCTION])
        finally:
            _TRACE.reset(tok)
        final.update({"content": content, "tool_calls": tool_calls})
        gcalls = [tc for tc in tool_calls or [] if tc.get("function", {}).get("name") == GRAPH_TOOL_NAME]
        if gcalls:
            args = gcalls[0]["function"].get("arguments")
            final["graph"], final["source"] = _clean_graph(_graph_args(args)), "tool_call"
            EV.tool_call(GRAPH_TOOL_NAME, args, result="accepted" if final["graph"] else "rejected: no root_causes")
        else:                                        # upstream's fallback: JSON parsed from the reply text
            final["graph"], final["source"] = _clean_graph(_parse_json(content or "")), "text"
    graph = canonical_graph(final["graph"], instances)
    EV.artifact("case_final", final)
    EV.artifact("case_graph", graph)
    trajectory, calls_per_trace = _trajectory()
    meta.update({"entries": entries, "results": results, "missing_results": missing,
                 "final": {k: v for k, v in final.items() if k != "prompt"}, "final_prompt_chars":
                     len(final["prompt"] or ""), "edge_source": "v2_case_final_causal_graph", "traces": traces,
                 "llm_calls_per_trace": calls_per_trace, "llm_calls": len(CALLS), "llm_failures": FAILURES,
                 "max_agent_parallel": int(os.environ["MAX_AGENT_PARALLEL"]), "elapsed_s": round(time.time() - t0, 1),
                 "log_result_guard": {"budget_chars": LOG_RESULT_CHAR_BUDGET, "bounded": list(LOG_GUARD)}})
    return {"graph": graph, "trajectory": trajectory, "meta": meta}


def main() -> None:
    shared_llm.load_env(HERE)
    tracker = install_hooks()
    payload = json.loads(sys.stdin.read())
    data_dir = Path(payload["data_dir"])
    cases.use(data_dir)                             # RCABench or RCA100, from the sandbox's tables
    cases.set_question(payload.get("question", ""))
    case_id = EV.new_id("case_")
    with EV.scope(role="rclv2", parent=case_id):
        EV.artifact("case_start", {"data_dir": str(data_dir), "case_id": case_id,
                                   "dataset": cases.active(), "alarm_endpoints": cases.alarm_endpoints()})
        res = run_case(data_dir, tracker)
        EV.artifact("final_graph", res["graph"])
    result = {
        "output": json.dumps(res["graph"], ensure_ascii=False),
        "trajectory": res["trajectory"],
        "usage": tracker.get_usage(),
        "rclagent_v2": res["meta"],
    }
    _STDOUT.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    _STDOUT.flush()


if __name__ == "__main__":
    main()

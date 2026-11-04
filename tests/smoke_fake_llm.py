#!/usr/bin/env python
"""Offline tests of the RCLAgent-V2 port: conversion checks + end-to-end run with a fake LLM server.

    agents/_rca_shared/.venv/bin/python agents/rclagent_v2/tests/smoke_fake_llm.py [case_converted_dir]
    agents/_rca_shared/.venv/bin/python agents/rclagent_v2/tests/smoke_fake_llm.py --dataset rca100 [<RCA100_CONV>/t001]

RCA100: the case runs from a harness-like sandbox (the 11 tables only, random directory name) with the t001
question (``_rca_shared/tests/rca100_case.py``); the fake blames payment through checkout, and the prompt
check expects upstream's re2ob system prompt with the case's services and the OTel Demo system name.

The fake LLM is a local HTTP server speaking the OpenAI chat-completions protocol (V2's llm.py posts to it
with ``requests``); OPENAI_BASE_URL points the agent at it, so no real LLM API is reachable. Tool results
come from upstream tool_server.py over HTTP. If the upstream clone with sample_data/ is available
(RCLV2_SAMPLE_DATA), upstream's coordinator is also run on it to count LLM calls per error trace.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
PY = Path(sys.executable)
DEFAULT_CASE = (Path(os.environ.get("RCABENCH_ROOT", AGENT_DIR.parents[1] / "datasets" / "RCABench-500"))
                / "ts0-mysql-container-kill-9t6n24" / "converted")
SAMPLE_DATA = Path(os.environ.get(
    "RCLV2_SAMPLE_DATA",
    "/tmp/RCLAgent-V2/sample_data"))       # a clone of upstream RCLAgent-V2 at UPSTREAM_COMMIT
QUESTION = """The following API endpoints are experiencing possible SLO violations and need investigation:
- HTTP GET http://ts-ui-dashboard:8080/api/v1/trainservice/trains

Please investigate the root cause of these SLO violations.
The telemetry data is stored in: `{d}`"""
SPAN_COLUMNS = ["timestamp", "cmdb_id", "span_id", "trace_id", "duration", "type", "status_code",
                "operation_name", "parent_span", "repeat_count"]
FAKE_GRAPH = {  # the fake's case-level graph (pod names and a self-loop exercise canonicalisation)
    "nodes": [{"component": "mysql-0", "state": ["UNAVAILABLE"]},
              {"component": "ts-auth-service", "state": ["HIGH_ERROR_RATE"]},
              {"component": "ts-ui-dashboard-64f6f55bb5-gv7sr", "state": ["HIGH_ERROR_RATE"]}],
    "edges": [{"source": "mysql-0", "target": "ts-auth-service"},
              {"source": "ts-auth-service", "target": "ts-ui-dashboard-64f6f55bb5-gv7sr"},
              {"source": "ts-ui-dashboard", "target": "ts-ui-dashboard"}],
    "root_causes": [{"component": "mysql-0", "state": ["UNAVAILABLE"]}],
    "component_to_service": {}}
EXPECTED_GRAPH = {
    "nodes": [{"component": "mysql", "state": ["UNAVAILABLE"]},
              {"component": "ts-auth-service", "state": ["HIGH_ERROR_RATE"]},
              {"component": "ts-ui-dashboard", "state": ["HIGH_ERROR_RATE"]}],
    "edges": [{"source": "mysql", "target": "ts-auth-service"},
              {"source": "ts-auth-service", "target": "ts-ui-dashboard"}],
    "root_causes": [{"component": "mysql", "state": ["UNAVAILABLE"]}],
    "component_to_service": {}}
# RCA100 (--dataset rca100): same shapes with OTel Demo names (pod names and a self-loop again)
RCA100_FAKE_GRAPH = {
    "nodes": [{"component": "payment-696d688896-klnbs", "state": ["HIGH_ERROR_RATE"]},
              {"component": "checkout", "state": ["HIGH_ERROR_RATE"]},
              {"component": "frontend-86465b665-chd5b", "state": ["HIGH_ERROR_RATE"]}],
    "edges": [{"source": "payment-696d688896-klnbs", "target": "checkout"},
              {"source": "checkout", "target": "frontend-86465b665-chd5b"},
              {"source": "frontend", "target": "frontend"}],
    "root_causes": [{"component": "payment-696d688896-klnbs", "state": ["HIGH_ERROR_RATE"]}],
    "component_to_service": {}}
RCA100_EXPECTED_GRAPH = {
    "nodes": [{"component": "payment", "state": ["HIGH_ERROR_RATE"]},
              {"component": "checkout", "state": ["HIGH_ERROR_RATE"]},
              {"component": "frontend", "state": ["HIGH_ERROR_RATE"]}],
    "edges": [{"source": "payment", "target": "checkout"},
              {"source": "checkout", "target": "frontend"}],
    "root_causes": [{"component": "payment", "state": ["HIGH_ERROR_RATE"]}],
    "component_to_service": {}}
LOG_BUDGET = 2000                                         # RCLV2_LOG_RESULT_CHAR_BUDGET of both V2 configs
# the service whose spans the fake marks abnormal, and the root it names
DATASET = os.environ.get("RCLV2_SMOKE_DATASET", "rcabench")
BLAME = {"rcabench": ("ts-auth-service", "mysql"), "rca100": ("frontend", "payment")}


def _blamed(cmdb_id: str) -> bool:
    """The span's service (cmdb_id = pod <service>-<hash>-<hash>, else the service) is the blamed one."""
    return re.sub(r"-[a-z0-9]{6,10}-[a-z0-9]{5}$", "", cmdb_id) == BLAME[DATASET][0]



# ------------------------------------------------------------------ fake OpenAI-compatible server
class FakeLLM(BaseHTTPRequestHandler):
    counts: Counter = Counter()
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs, tools = body["messages"], [t["function"]["name"] for t in body.get("tools") or []]
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        pod = (re.findall(r'"cmdb_id":\s*"([^"]+)"', user) or ["unknown"])[0]
        ts = (re.findall(r'"timestamp":\s*(\d+)', user) or ["0"])[0]
        n_tool = sum(1 for m in msgs if m["role"] == "tool")
        calls, content = None, ""
        if "output_causal_graph" in tools:
            kind = "final"
            calls = [("output_causal_graph", RCA100_FAKE_GRAPH if DATASET == "rca100" else FAKE_GRAPH)]
        elif "search_logs" in tools and n_tool == 0:
            kind = "verify_tools"
            calls = [("search_logs", {"service_name": pod, "timestamp": ts}),
                     ("search_fluctuating_metrics", {"service_name": pod, "timestamp": ts})]
        elif "search_logs" in tools:
            kind = "verify_conclusion"
            root = BLAME[DATASET][1]                          # e.g. the fake blames mysql via ts-auth-service
            auth = _blamed(pod)
            content = json.dumps({"span_id": "x", "service_name": pod, "is_abnormal": auth,
                                  "has_error_logs": auth, "has_metric_anomaly": False,
                                  "key_symptoms": f"{root} errors in logs" if auth else None,
                                  "hypothesis": "fake"})
        else:
            kind = "consolidation"
            root = BLAME[DATASET][1]
            content = json.dumps({"span_id": "x", "service_name": pod,
                                  "local_root_cause": root if _blamed(pod) else pod,
                                  "has_error_logs": _blamed(pod), "reason": "fake",
                                  "confidence": 0.5})
        with self.lock:
            self.counts[kind] += 1
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                  "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
        out = json.dumps({"id": "fake", "object": "chat.completion", "model": body.get("model"),
                          "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def start_fake() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


# ------------------------------------------------------------------ conversion checks
def check_compaction_unit() -> None:
    """Span compaction on a hand-made trace: A (INTERNAL re-attach), B (identical sibling subtrees)."""
    import pandas as pd

    sys.path.insert(0, str(AGENT_DIR))
    import native_data

    rows = [  # span_id, parent, type, service, name, status, duration
        ("r", None, "CLIENT", "lg", "GET /x", "error", 100),
        ("s", "r", "SERVER", "a", "GET /x", "0", 90),
        ("i", "s", "INTERNAL", "a", "Ctl.x", "0", 80),           # A: dropped, children move to s
        ("i2", "i", "INTERNAL", "a", "Svc.x", "0", 70),          # A: nested, dropped
        ("c1", "i2", "CLIENT", "a", "POST", "0", 10),            # POST -> b
        ("b1", "c1", "SERVER", "b", "POST /b", "0", 9),
        ("c2", "i2", "CLIENT", "a", "POST", "0", 30),            # POST -> b: same subtree as c1 unless db
        ("b2", "c2", "SERVER", "b", "POST /b", "0", 29),
        ("c3", "i2", "CLIENT", "a", "POST", "0", 5),             # same name, other callee: kept
        ("d3", "c3", "SERVER", "c", "POST /c", "0", 4),
        ("c4", "i2", "CLIENT", "a", "POST", "error", 50),        # error status: kept apart
        ("b4", "c4", "SERVER", "b", "POST /b", "error", 49),
        ("db", "b1", "CLIENT", "b", "SELECT t", "0", 1),         # makes c1's subtree differ from c2's
    ]
    df = pd.DataFrame(rows, columns=["span_id", "parent_span", "type", "service", "operation_name",
                                     "status_code", "duration"])
    out, st = native_data.compact_spans(df)
    kept = dict(zip(out.span_id, out.parent_span))
    assert st["internal_dropped"] == 2 and st["sibling_merged_dropped"] == 0, st
    assert set(kept) == set(df.span_id) - {"i", "i2"}, kept
    assert kept["c1"] == kept["c2"] == kept["c3"] == kept["c4"] == "s", kept
    out2, st2 = native_data.compact_spans(df[df.span_id != "db"])          # now c1 and c2 are identical
    k2 = dict(zip(out2.span_id, out2.repeat_count))
    assert "c1" not in k2 and "b1" not in k2 and k2["c2"] == 2, k2        # the longer one stays, count 2
    assert k2["c3"] == 1 and k2["c4"] == 1 and k2["r"] == 1 and st2["sibling_merged_dropped"] == 2
    print("[compaction unit] OK")


def check_log_guard_unit() -> None:
    """RCLV2_LOG_RESULT_CHAR_BUDGET: results within the budget unchanged; identical lines of one pod merged
    with repeat_count; rows cut with a note when the merged result is still too large."""
    import io

    import pandas as pd

    sys.path.insert(0, str(AGENT_DIR))
    import agent_runner
    sys.stdout = sys.__stdout__
    cols = ["log_id", "timestamp", "cmdb_id", "log_name", "value"]
    rows = [(f"x_{i}", 100 + i, "frontend-1-a", "log_frontend",      # same template: numbers and ids vary
             f"severity: unknown, message: Error: boom {i} at 10.0.1.{i} id=a1b2c3{i:04d}") for i in range(50)]
    rows += [("y_0", 200, "frontend-1-a", "log_frontend", "severity: error, message: other"),
             ("y_1", 201, "frontend-2-b", "log_frontend", "severity: error, message: other")]   # another pod
    text = pd.DataFrame(rows, columns=cols).to_csv(index=False)
    assert agent_runner.bound_log_result(text, len(text)) == (text, None)
    out, rec = agent_runner.bound_log_result(text, len(text) - 1)
    df = pd.read_csv(io.StringIO(out))
    assert list(df.log_id) == ["x_0", "y_0", "y_1"] and list(df.repeat_count) == [50, 1, 1], df
    assert df.value[0] == rows[0][4], "the first occurrence is kept verbatim"
    assert rec["rows"] == 52 and rec["distinct_rows"] == rec["kept_rows"] == 3
    many = [(f"z_{i}", i, "p", "log_p", f"severity: {'error' if i % 50 == 7 else 'warn'}, message: line {'x' * (i % 26)}"
             + chr(65 + i % 26) * 40) for i in range(200)]
    out, rec = agent_runner.bound_log_result(pd.DataFrame(many, columns=cols).to_csv(index=False), 2000)
    body = out.rstrip().splitlines()
    kept = pd.read_csv(io.StringIO("\n".join(body[:-1])))
    assert len(out) <= 2000 and body[-1].startswith(f"# {rec['distinct_rows'] - rec['kept_rows']} more distinct lines")
    assert kept.value.str.startswith("severity: error").sum() == min(4, len(kept)), "error-like lines are kept first"
    assert list(kept.timestamp) == sorted(kept.timestamp), "kept rows stay in time order"
    long = pd.DataFrame([("w_0", 1, "p", "log_p", "severity: warn, message: " + "y" * 9000)], columns=cols)
    out, _ = agent_runner.bound_log_result(long.to_csv(index=False), 2000)
    assert len(out) <= 2000 and "…" in out
    print(f"[log guard unit] OK: template merge 52 -> 3 rows; cut keeps {rec['kept_rows']} of "
          f"{rec['distinct_rows']} distinct lines, errors first, in time order; long line cut")


def check_compaction_paths(case: Path, rep: dict) -> None:
    """On the case's error traces: B keeps exactly the set of root-to-span (service, name, status) paths
    of the trace after A (recomputed here), and never touches the roots."""
    import duckdb

    sys.path.insert(0, str(AGENT_DIR))
    import native_data

    tids = [r["trace_id"] for r in rep["error_traces"]["rows"]]
    if DATASET == "rca100":                    # RCA100's own columns: OTel kind / status codes
        raw = duckdb.sql(f"""SELECT spanId AS span_id, nullif(parentSpanId, '') AS parent_span,
            CASE kind WHEN '1' THEN 'INTERNAL' WHEN '2' THEN 'SERVER' WHEN '3' THEN 'CLIENT'
                      WHEN '4' THEN 'PRODUCER' WHEN '5' THEN 'CONSUMER' ELSE 'UNSPECIFIED' END AS "type",
            serviceName AS service, spanName AS operation_name,
            CASE WHEN statusCode = '2' THEN 'error' ELSE '0' END AS status_code, duration_ns AS duration
            FROM read_parquet(['{case}/normal_traces.parquet', '{case}/abnormal_traces.parquet'])
            WHERE traceId IN (SELECT unnest(?))""", params=[tids]).df()
    else:
        raw = duckdb.sql(f"""SELECT span_id, nullif(parent_span_id, '') AS parent_span,
            upper("attr.span_kind") AS "type", service_name AS service, span_name AS operation_name,
            CASE WHEN "attr.status_code" = 'Error' THEN 'error' ELSE '0' END AS status_code, duration
            FROM read_parquet(['{case}/normal_traces.parquet', '{case}/abnormal_traces.parquet'])
            WHERE trace_id IN (SELECT unnest(?))""", params=[tids]).df()
    ac = native_data._drop_reattach(raw, raw.type.eq("INTERNAL") & raw.parent_span.notna())
    comp = native_data.compact_spans(raw)[0]

    def paths(df, root):
        lab = {s: (a, b, c) for s, a, b, c in zip(df.span_id, df.service, df.operation_name, df.status_code)}
        kids = df.groupby("parent_span")["span_id"].apply(list).to_dict()
        acc, stack = set(), [(root, (lab[root],))]
        while stack:
            s, p = stack.pop()
            acc.add(p)
            stack += [(c, p + (lab[c],)) for c in kids.get(s, [])]
        return acc

    for r in rep["error_traces"]["rows"]:
        assert paths(ac, r["span_id"]) == paths(comp, r["span_id"]), r["span_id"]
    e = rep["error_traces"]
    assert all(a <= b for a, b in zip(e["tree_sizes"], e["tree_sizes_uncompacted"]))
    print(f"[compaction paths] OK: spans/tree {e['tree_sizes_uncompacted']} -> {e['tree_sizes']}")


def check_conversion(case: Path, tmp: Path) -> dict:
    import pandas as pd

    sys.path.insert(0, str(AGENT_DIR))
    import native_data

    rep = native_data.convert(case, tmp / "native")
    assert native_data.convert(case, tmp / "native")["cached"] is True
    root = Path(rep["root"])
    assert not (root / "groundtruth.csv").exists(), "no ground truth in the agent-visible dir"
    env = json.loads((case / "env.json").read_text())
    spans = pd.read_csv(root / "trace/all/trace_jaeger-span.csv", dtype={"status_code": str})
    assert list(spans.columns) == SPAN_COLUMNS
    assert spans.timestamp.between(int(env["NORMAL_START"]) * 1000, (int(env["ABNORMAL_END"]) + 60) * 1000).all()
    assert set(spans.status_code) <= {"0", "error"}
    assert (spans.repeat_count >= 1).all() and rep["span_compaction"]["enabled"]
    assert not (spans.type.eq("INTERNAL") & spans.parent_span.notna()).any()
    m = pd.read_csv(root / "metric/all/metrics.csv")
    assert list(m.columns) == ["timestamp", "node_id", "service_name", "kpi_name", "value"]
    assert (m.timestamp % native_data.BUCKET_S == 0).all()
    assert m.fillna("").groupby(["timestamp", "node_id", "service_name", "kpi_name"]).size().max() == 1
    per_series = m.dropna(subset=["node_id", "service_name"]).groupby(["kpi_name", "node_id", "service_name"]).size()
    assert per_series.max() >= 40, per_series.max()
    assert m.service_name.fillna("").str.startswith("mysql").any()
    sn = pickle.load(open(root / "metric/service_node_map.pkl", "rb"))
    ns = pickle.load(open(root / "metric/node_service_map.pkl", "rb"))
    assert sn and all(p in ns[n] for p, n in sn.items()) and any(p.startswith("mysql") for p in sn)
    logs = pd.read_csv(root / "log/all/logs.csv")
    assert list(logs.columns) == ["log_id", "timestamp", "cmdb_id", "log_name", "value"]
    assert logs.value.str.match(r"severity: \w+, message: ").all()
    assert not logs.value.str.lower().str.contains(r"severity:\s*(?:info|debug)").any()
    lines = (root / "error_traces.txt").read_text().splitlines()
    e = rep["error_traces"]
    assert e["rule"] == "nezha_top5_abnormal_window" and not e["fallback"]
    assert len(lines) == 1 + e["kept"] and e["selected"] == min(5, e["window_roots"])
    assert e["kept"] + len(e["duplicate_trees_dropped"]) == e["selected"]
    a0, a1 = int(env["ABNORMAL_START"]) * 1000, int(env["ABNORMAL_END"]) * 1000
    roots = spans[spans.parent_span.isna() & spans.timestamp.between(a0, a1)]
    expected = set(roots.nlargest(5, "duration").span_id)       # upstream Nezha rule on the abnormal window
    got = [ln.split() for ln in lines[1:]]
    assert {p[3] for p in got} | set(e["duplicate_trees_dropped"]) == expected, (got, expected)
    ts = [int(p[1]) for p in got]
    assert ts == sorted(ts) and [int(p[0]) for p in got] == list(range(len(got)))
    print(f"[conversion] OK: {native_data.summary(rep)}")
    return rep


def check_conversion_rca100(case: Path, tmp: Path) -> dict:
    """RCA100 sandbox -> V2 layout; the abnormal window is the question's alert window."""
    import pandas as pd

    sys.path.insert(0, str(AGENT_DIR))
    import native_data
    import rca100_case

    rep = native_data.convert(case, tmp / "native")
    assert native_data.convert(case, tmp / "native")["cached"] is True
    root = Path(rep["root"])
    assert not (root / "groundtruth.csv").exists() and not (case / "task.json").exists()
    spans = pd.read_csv(root / "trace/all/trace_jaeger-span.csv", dtype={"status_code": str})
    assert list(spans.columns) == SPAN_COLUMNS
    assert set(spans.status_code) <= {"0", "error"} and (spans.status_code == "error").any()
    assert set(spans.type) <= {"SERVER", "CLIENT", "PRODUCER", "CONSUMER", "INTERNAL"}, set(spans.type)
    assert not (spans.type.eq("INTERNAL") & spans.parent_span.notna()).any()
    assert spans.cmdb_id.str.match(r"^payment-[a-z0-9]+-[a-z0-9]{5}$").any(), "cmdb_id = pod"
    m = pd.read_csv(root / "metric/all/metrics.csv")
    assert list(m.columns) == ["timestamp", "node_id", "service_name", "kpi_name", "value"]
    assert (m.timestamp % native_data.BUCKET_S == 0).all() and m.kpi_name.str.startswith("apm.").all()
    assert m.fillna("").groupby(["timestamp", "node_id", "service_name", "kpi_name"]).size().max() == 1
    cont = m.dropna(subset=["node_id", "service_name"])        # the rows upstream's metric tool tests
    assert cont.service_name.str.startswith("payment-").any() and cont.node_id.str.startswith("cn-hongkong.").all()
    assert cont.groupby(["kpi_name", "node_id", "service_name"]).size().max() >= 40
    sn = pickle.load(open(root / "metric/service_node_map.pkl", "rb"))
    assert sn and all(p.split("-")[0] and n.startswith("cn-hongkong.") for p, n in sn.items())
    logs = pd.read_csv(root / "log/all/logs.csv")
    assert list(logs.columns) == ["log_id", "timestamp", "cmdb_id", "log_name", "value"]
    assert logs.value.str.match(r"severity: \w+, message: ").all()
    assert not logs.value.str.lower().str.contains(r"severity:\s*(?:info|debug)").any()
    assert logs.value.str.contains(r"^severity: warn, message: \[k8s event\] Warning ", regex=True).any()
    assert logs.log_name.eq("log_payment").any() and not logs.log_name.eq("log_cart").any(), "cart logs are INFO"
    e = rep["error_traces"]
    a0, a1 = (t * 1000 for t in rca100_case.ALERT_WINDOW)
    assert e["window_source"] == "question alert window" and e["window_ms"] == [a0, a1], e["window_ms"]
    assert e["rule"] == "nezha_top5_abnormal_window" and not e["fallback"]
    lines = (root / "error_traces.txt").read_text().splitlines()
    assert len(lines) == 1 + e["kept"] and e["kept"] + len(e["duplicate_trees_dropped"]) == e["selected"] == 5
    roots = spans[spans.parent_span.isna() & spans.timestamp.between(a0, a1)]
    assert {ln.split()[3] for ln in lines[1:]} | set(e["duplicate_trees_dropped"]) == set(
        roots.nlargest(5, "duration").span_id)
    print(f"[conversion rca100] OK: {native_data.summary(rep)}")
    return rep


# ------------------------------------------------------------------ end-to-end
def run_runner(case: Path, tmp: Path, base_url: str, question: str | None = None) -> tuple[dict, Path]:
    events = tmp / "run"
    env = {**os.environ, "RCA_EVENTS_DIR": str(events), "OPENAI_API_KEY": "fake-key", "OPENAI_BASE_URL": base_url,
           "RCA_MODEL": "qwen3.5-plus", "RCA_ENV_FILE": str(tmp / "no.env"), "RCLV2_MAX_AGENT_PARALLEL": "8"}
    env.pop("RCLV2_NATIVE_DIR", None)
    env["RCLV2_LOG_RESULT_CHAR_BUDGET"] = str(LOG_BUDGET)  # as configs/agents/rclagent-v2-*.yaml
    payload = {"question": question or QUESTION.format(d=case), "system_prompt": "SYS", "user_prompt": "USR",
               "compress_system_prompt": "CSYS", "compress_user_prompt": "CUSR", "data_dir": str(case)}
    p = subprocess.run([str(PY), str(AGENT_DIR / "agent_runner.py")], cwd=AGENT_DIR,
                       input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=3600)
    (tmp / "run.stderr.log").write_text(p.stderr)
    assert p.returncode == 0, f"exit {p.returncode}\n{p.stderr[-3000:]}"
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout must hold exactly the result line, got {len(lines)}"
    return json.loads(lines[0]), events / "events.jsonl"


def check_run(res: dict, events_file: Path, rep: dict) -> None:
    assert set(res) == {"output", "trajectory", "usage", "rclagent_v2"}, set(res)
    graph = json.loads(res["output"])
    expected = RCA100_EXPECTED_GRAPH if DATASET == "rca100" else EXPECTED_GRAPH
    assert graph == expected, graph                           # the model's case graph, canonicalised
    meta = res["rclagent_v2"]
    n = meta["n_traces"]
    assert n == rep["error_traces"]["kept"] and not meta["missing_results"] and not meta["llm_failures"]
    ev = [json.loads(ln) for ln in events_file.read_text().splitlines()]
    kinds = Counter(e["event"] for e in ev)
    all_llm = [e for e in ev if e["event"] == "llm_call"]
    llm = [e for e in all_llm if e["role"].startswith("rclv2:")]          # V2's own calls (llm.py)
    assert len(llm) == meta["llm_calls"] and len(all_llm) == res["usage"]["llm_call_count"], \
        (len(llm), meta["llm_calls"], len(all_llm), res["usage"])
    assert all(ROLE_RE.match(e["role"]) for e in llm), Counter(e["role"] for e in llm).most_common(3)
    roles = Counter(phase_of(e) for e in llm)
    assert set(roles) == {"verification", "consolidation", "final"}, roles
    assert roles["final"] == 1, "one case-level final call; no per-trace final call"
    assert all(e["parent_id"] and (e["parent_id"].startswith("trace-") or phase_of(e) == "final") for e in llm)
    check_case_final(res, ev, llm, rep)
    check_prompt(ev, llm)
    tools = Counter(e["tool"] for e in ev if e["event"] == "tool_call")
    assert set(tools) == {"search_span", "search_traces", "search_logs", "search_fluctuating_metrics",
                          "output_causal_graph"}, tools
    assert tools["search_span"] == n and tools["output_causal_graph"] == 1
    sfm = [e for e in ev if e["event"] == "tool_call" and e["tool"] == "search_fluctuating_metrics"]
    flagged = [e for e in sfm if e["result"] and e["result"].startswith("key,")]
    # metric windows: regular = normal window, current = abnormal window, so the two means differ
    rows = [ln.split(",") for e in flagged for ln in e["result"].splitlines()[1:]]
    if DATASET == "rcabench":
        assert rows and any(r[1] != r[3] for r in rows), "regular_mean must not equal current_mean everywhere"
    else:                                               # RCA100: pods of the frontend-proxy/frontend traces
        assert all(e["result"] for e in sfm) and all(r[1] != r[3] for r in rows), rows[:3]
    guard = meta["log_result_guard"]
    logs_res = [e["result"] for e in ev if e["event"] == "tool_call" and e["tool"] == "search_logs"]
    assert guard["budget_chars"] == LOG_BUDGET and guard["bounded"], guard   # e.g. t001's frontend pod
    assert all(len(r) <= LOG_BUDGET for r in logs_res if isinstance(r, str))
    assert any("repeat_count" in r.splitlines()[0] for r in logs_res if isinstance(r, str) and r)
    arts = Counter(e["name"] for e in ev if e["event"] == "artifact")
    for a in ("rclv2_native_data", "metric_windows", "error_traces", "known_services_injected", "trace_result_file",
              "case_final", "case_graph", "final_graph"):
        assert arts[a], arts
    agents = Counter(m["agent"] for m in res["trajectory"])
    assert all(a.startswith("rclv2:trace-") or a == "rclv2:case:final" for a in agents), agents
    sizes = rep["error_traces"]["tree_sizes"]
    per_trace = meta["llm_calls_per_trace"]
    for k, sz in enumerate(sizes):                         # fake: 2 verification calls + 1 consolidation per span
        assert per_trace[str(k)] == 3 * sz, (k, sz, per_trace)
    assert per_trace["case"] == 1
    print(f"[run] OK: graph={graph} traces={n} tree_sizes={sizes} llm_calls={per_trace} events={dict(kinds)} "
          f"tools={dict(tools)} flagged_metric_queries={len(flagged)}/{len(sfm)} usage={res['usage']}")


ROLE_RE = re.compile(r"^rclv2:(trace-\d+:span-[0-9a-f]+:(verification|consolidation|fallback)|case:final)$")


def phase_of(e: dict) -> str:
    return e["role"].rsplit(":", 1)[1]


def check_case_final(res: dict, ev: list, llm: list, rep: dict) -> None:
    """One final call per case over every trace's evidence blocks; per-trace artifacts saved."""
    meta = res["rclagent_v2"]
    n = meta["n_traces"]
    sys.path.insert(0, str(AGENT_DIR))
    import agent_runner
    sys.stdout = sys.__stdout__
    final = [e for e in llm if phase_of(e) == "final"][0]
    msgs = final["request"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert [t["function"]["name"] for t in final["request"]["tools"]] == ["output_causal_graph"]
    prompt = msgs[1]["content"]
    case = next(e["data"] for e in ev if e["event"] == "artifact" and e["name"] == "case_final")
    assert case["prompt"] == prompt and case["source"] == "tool_call" and case["tool_calls"]
    reports = {e["data"]["k"]: e["data"] for e in ev if e["event"] == "artifact" and e["name"] == "trace_report"}
    assert sorted(reports) == list(range(n))
    pos = []
    for k in range(n):                                  # every trace's evidence blocks, verbatim, in order
        r = reports[k]
        up = r["upstream_final_prompt_not_sent"]
        block = up[up.index(agent_runner.EVIDENCE_START):up.index(agent_runner.EVIDENCE_END)]
        header = f"=== Trace {k + 1}/{n} (root operation: {rep['error_traces']['rows'][k]['operation_name']}) ===\n"
        assert header + block in prompt, k
        assert r["root_level_report"] and r["root_level_report"] in block and r["geg_summary"] in block
        pos.append(prompt.index(header))
    assert pos == sorted(pos)
    assert prompt.startswith(agent_runner.CASE_OPENER)
    tail = "\n\n" + agent_runner.GOAL_LINE + "\n\n" + agent_runner.GRAPH_INSTRUCTIONS
    assert tail in prompt and "print_results" not in prompt
    for t in meta["traces"]:
        assert json.loads(open(Path(meta["native_root"]) / "result" /
                               f"conversation_trace_{t['k'] + 1}.txt").read())["root_causes"] == []
        tree = {x["span_id"]: x for x in t["tree"]}
        assert len(tree) == rep["error_traces"]["tree_sizes"][t["k"]]
        assert set(t["verdicts"]) == set(tree) and set(t["consolidations"]) == set(tree) and t["root_level_report"]
    arts = Counter(e["name"] for e in ev if e["event"] == "artifact")
    for a in ("trace_tree", "span_verdicts", "span_consolidations", "trace_report", "trace_result_file"):
        assert arts[a] == n, (a, arts[a])
    raw = final["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert _json_eq(raw, json.dumps(RCA100_FAKE_GRAPH if DATASET == "rca100" else FAKE_GRAPH)) and _json_eq(case["tool_calls"][0]["function"]["arguments"], raw)
    fb = agent_runner._clean_graph(agent_runner._parse_json(      # text fallback (no tool call)
        'Here: {"root_causes": [{"component": "mysql", "state": ["UNAVAILABLE"]}], "nodes": [], "edges": []} done'))
    assert fb and fb["root_causes"][0]["component"] == "mysql" and agent_runner._clean_graph({"nodes": []}) is None
    print(f"[case_final] OK: 1 final call, prompt {len(prompt)} chars with {n} trace blocks")


def _json_eq(a, b) -> bool:
    return json.loads(a) == json.loads(b) if isinstance(a, str) and isinstance(b, str) else a == b


def check_prompt(ev: list, llm: list) -> None:
    """Patched system prompt (nezha for RCABench, re2ob for RCA100): in every LLM call, differs from upstream
    only in the service block (and, for re2ob, the system name)."""
    import difflib
    sys.path.insert(0, str(AGENT_DIR))
    import agent_runner
    sys.stdout = sys.__stdout__
    art = next(e["data"] for e in ev if e["event"] == "artifact" and e["name"] == "system_prompt_patched")
    patched, upstream = art["prompt"], art["upstream_prompt"]
    injected = next(e["data"] for e in ev if e["event"] == "artifact" and e["name"] == "known_services_injected")
    art["services"] = injected["services"]
    dtype = agent_runner.DATASET_TYPES[DATASET]
    assert injected["dataset_type"] == dtype
    for e in llm:
        sysmsg = e["request"]["messages"][0]
        assert sysmsg["role"] == "system" and sysmsg["content"] == patched and BLAME[DATASET][1] in sysmsg["content"], e["seq"]
    start, end = agent_runner.SERVICE_BLOCKS[dtype]
    ref = upstream
    if dtype in agent_runner.SYSTEM_NAMES:                 # re2ob: the system name is replaced too
        old, new = agent_runner.SYSTEM_NAMES[dtype]
        assert ref.count(old) == 1 and new in patched and old not in patched
        ref, start = ref.replace(old, new), start.replace(old, new)
    i, j = ref.index(start) + len(start), ref.index(end)
    old_block = ref[i:j]
    k = patched.index(end)
    new_block = patched[i:k]
    old_lines, new_lines = old_block.split("\n"), new_block.split("\n")
    width = max(map(len, old_lines))
    assert dtype != "nezha" or width == agent_runner.SERVICE_BLOCK_WIDTH
    for lines in (old_lines, new_lines):                   # same layout rules as upstream's hand-wrapped block
        assert all(ln.startswith("  ") and not ln.startswith("   ") for ln in lines)
        assert all(ln.endswith(",") for ln in lines[:-1]) and not lines[-1].endswith(",")
        assert max(map(len, lines)) <= width
    assert [n.strip() for n in new_block.replace("\n", " ").split(",")] == art["services"]
    assert patched[:i] == ref[:i] and patched[k:] == ref[j:], "only the service block may differ"
    diff = [ln for ln in difflib.unified_diff(upstream.splitlines(), patched.splitlines(), lineterm="", n=0)
            if not ln.startswith(("---", "+++"))]
    print("[system_prompt] OK: patched prompt in all %d LLM calls; diff vs upstream:\n%s" % (len(llm), "\n".join(diff)))


def check_known_services(case: Path, meta: dict) -> None:
    """Injected list = the case's services; what upstream _normalise_name does with pod names."""
    sys.path.insert(0, str(AGENT_DIR.parent))
    from _rca_shared import cases
    services = cases.instances_of(case)
    if DATASET == "rca100":
        return check_known_services_rca100(services, meta)
    assert meta["known_services"] == services and "mysql" in services and "loadgenerator" not in services
    code = f"""
import json, sys
sys.path.insert(0, {str(AGENT_DIR / 'upstream')!r}); sys.path.insert(0, {str(AGENT_DIR)!r})
import config
config.KNOWN_SERVICES['nezha'] = {services!r}
import coordinator, agent_runner
names = ['mysql-0', 'ts-auth-service-5dd97d5ccd-f7nlt', 'MySQL', 'ts-auth-service']
sys.__stdout__.write(json.dumps({{n: [coordinator._normalise_name(n), agent_runner._to_service(coordinator._normalise_name(n), {services!r})]
                  for n in names}}) + chr(10))
"""
    env = {**os.environ, "DATASET_TYPE": "nezha", "LLM_API_KEY": "fake"}
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    # upstream _normalise_name keeps valid pod names (it only repairs hyphenation); the port's
    # aggregation maps them to services
    assert out["mysql-0"] == ["mysql-0", "mysql"], out
    assert out["ts-auth-service-5dd97d5ccd-f7nlt"] == ["ts-auth-service-5dd97d5ccd-f7nlt", "ts-auth-service"], out
    assert out["MySQL"] == ["mysql", "mysql"] and out["ts-auth-service"] == ["ts-auth-service", "ts-auth-service"], out
    print(f"[known_services] OK: {len(services)} services injected; name -> [_normalise_name, service]: {out}")


def check_known_services_rca100(services: list, meta: dict) -> None:
    assert meta["known_services"] == services and meta["dataset_type"] == "re2ob"
    assert "payment" in services and "frontend-proxy" in services and "load-generator" not in services
    code = f"""
import json, sys
sys.path.insert(0, {str(AGENT_DIR / 'upstream')!r}); sys.path.insert(0, {str(AGENT_DIR)!r})
import config
config.KNOWN_SERVICES['re2ob'] = {services!r}
import coordinator, agent_runner
names = ['payment-696d688896-klnbs', 'frontend-proxy-67d64dc4f8-xz2fd', 'Payment', 'product-catalog']
sys.__stdout__.write(json.dumps({{n: [coordinator._normalise_name(n), agent_runner._to_service(coordinator._normalise_name(n), {services!r})]
                  for n in names}}) + chr(10))
"""
    env = {**os.environ, "DATASET_TYPE": "re2ob", "LLM_API_KEY": "fake"}
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["payment-696d688896-klnbs"] == ["payment-696d688896-klnbs", "payment"], out
    assert out["frontend-proxy-67d64dc4f8-xz2fd"] == ["frontend-proxy-67d64dc4f8-xz2fd", "frontend-proxy"], out
    assert out["Payment"] == ["payment", "payment"] and out["product-catalog"] == ["product-catalog"] * 2, out
    print(f"[known_services] OK (re2ob): {len(services)} services injected; name -> [_normalise_name, service]: {out}")


def sample_data_calls(tmp: Path, base_url: str) -> None:
    """Upstream coordinator on upstream's sample_data (not shipped): LLM calls per error trace, fake LLM."""
    if not (SAMPLE_DATA / "error_traces.txt").exists():
        print(f"[sample_data] skipped: {SAMPLE_DATA} not found")
        return
    data = tmp / "sample_data"
    shutil.copytree(SAMPLE_DATA, data)
    shutil.rmtree(data / "result", ignore_errors=True)
    code = f"""
import sys, threading, socket
sys.path.insert(0, {str(AGENT_DIR / 'upstream')!r})
import tool_server, coordinator
from werkzeug.serving import make_server
import config
srv = make_server('127.0.0.1', config.TOOL_SERVER_PORT, tool_server.app, threaded=True)
threading.Thread(target=srv.serve_forever, daemon=True).start()
coordinator.run_all()
srv.shutdown()
"""
    with __import__("socket").socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    before = sum(FakeLLM.counts.values())
    env = {**os.environ, "DATA_ROOT": str(data), "DATASET_TYPE": "aiops2022", "TOOL_SERVER_PORT": str(port),
           "LLM_API_URL": base_url + "/chat/completions", "LLM_API_KEY": "fake", "LLM_MODEL": "qwen3.5-plus"}
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, env=env, timeout=1800)
    assert p.returncode == 0, p.stderr[-2000:]
    n_traces = len((data / "error_traces.txt").read_text().splitlines()) - 1
    import pandas as pd
    spans = pd.read_csv(data / "trace/all/trace_jaeger-span.csv")
    calls = sum(FakeLLM.counts.values()) - before
    print(f"[sample_data] upstream coordinator: {n_traces} error trace(s), {len(spans)} spans in the trace file, "
          f"{calls} LLM calls with the fake (2 verification + 1 consolidation per span + 1 final)")


def main() -> None:
    global DATASET
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=None, help="RCABench converted/ dir, or RCA100 converted t001 dir")
    ap.add_argument("--dataset", choices=tuple(BLAME), default="rcabench")
    args = ap.parse_args()
    DATASET = args.dataset                                  # the in-process fake LLM reads it per request
    tmp = Path(tempfile.mkdtemp(prefix="rclv2_smoke_"))
    question = None
    if DATASET == "rca100":
        sys.path.insert(0, str(AGENT_DIR.parent / "_rca_shared" / "tests"))
        sys.path.insert(0, str(AGENT_DIR.parent))
        import rca100_case
        from _rca_shared import cases

        case, question = rca100_case.sandbox(rca100_case.converted_case(args.case), tmp), rca100_case.QUESTION
        cases.use(case)
        cases.set_question(question)                       # in-process conversion check: alert window
    else:
        case = Path(args.case) if args.case else DEFAULT_CASE
    print(f"[{DATASET}] case {case}; events/logs under {tmp}")
    srv, base_url = start_fake()
    try:
        check_compaction_unit()
        check_log_guard_unit()
        rep = check_conversion_rca100(case, tmp) if DATASET == "rca100" else check_conversion(case, tmp)
        check_compaction_paths(case, rep)
        res, events_file = run_runner(case, tmp, base_url, question)
        check_run(res, events_file, rep)
        check_known_services(case, res["rclagent_v2"])
        if DATASET == "rcabench":
            sample_data_calls(tmp, base_url)
    finally:
        srv.shutdown()
    print("ALL SMOKE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

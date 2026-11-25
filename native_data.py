"""Convert one RCABench or RCA100 case into RCLAgent-V2's canonical layout (upstream README, "Canonical Output Layout").

RCA100 (OpenTelemetry Demo) goes through the same rules below: ``_rca_shared/cases.py`` serves its
tables through the same canonical views (``_rca_shared/rca100.py``: OTel span kinds and status
codes, pod -> node from the topology, metric names ``<entity_set>.<metric>``, log levels read from
the line, Kubernetes events as log records). Its abnormal window is the question's alert window.

RCABench is TrainTicket, like V2's Nezha benchmark, so the mapping mirrors
``upstream/preprocess/preprocess_nezha.py`` wherever RCABench carries the same
information, and the AIOps 2022 canonical format (``preprocessing_*.py``) elsewhere.
Both windows (normal_* = reference, abnormal_* = incident) go into the data files;
no window label is written. Ground truth is never read and ``groundtruth.csv`` is
not written (only ``evaluate.py`` and Nezha's error-trace selection use it).

trace/all/trace_jaeger-span.csv   (timestamp, cmdb_id, span_id, trace_id, duration, type,
                                   status_code, operation_name, parent_span)
    timestamp       span start, epoch ms              (Nezha: StartTimeUnixNano // 1e6)
    cmdb_id         pod name, else service            (Nezha: PodName)
    duration        µs, the canonical AIOps unit that preprocessing_traces.py's
                    DURATION_THRESHOLD_US assumes     (Nezha keeps ns)
    type            span kind (SERVER/CLIENT/...)     (Nezha: constant "rpc", no kind)
    status_code     "0" for Ok/Unset (Nezha's ok value), "error" for Error spans
                                                      (Nezha: constant "0", no status)
    operation_name  span name
    parent_span     parent span id, empty for roots   (Nezha: "root" -> NaN)
    repeat_count    added column: how many identical sibling spans this row stands for (see below)
    loadgenerator kept (its CLIENT spans are the trace roots).
    Span compaction (``RCLV2_SPAN_COMPACTION``, default 1; 0 = every span as recorded). RCABench's
    OpenTelemetry auto-instrumentation records every in-process method call and every repeated DB/RPC
    call: the largest error trace of the 100-case set has 8,502 spans but only 43 distinct
    (service, span name) pairs, while upstream's AIOps sample trace has 26 spans. V2 gives every span a
    Dedicated Agent, so two data-level steps bring traces back to upstream's granularity:
      A  INTERNAL spans (in-process methods) that have a parent are dropped; their children are
         re-attached to the nearest ancestor that is kept. Upstream's datasets carry call spans only
         (rpc/http/db; caller and callee side of each call), no in-process methods.
      B  children of one parent whose whole subtrees are identical (same service, span name and status
         at every node, compared as sets of distinct child subtrees, bottom-up) are merged into one
         row: the longest one is kept with its subtree, ``repeat_count`` = group size; the others and
         their subtrees are dropped. The subtree is part of the key because a CLIENT span's name is
         often just "POST": its callee shows only in its SERVER child. Every distinct call path, every
         error status and the slowest instance stay.
    Root spans are never dropped or changed, so error-trace selection is unaffected.
metric/all/metrics.csv            (timestamp s, node_id, service_name, kpi_name, value)
    RCABench metrics (metrics ∪ metrics_sum, ~5 s) resampled to BUCKET_S (10 s): mean of
    all rows of a series key in the bucket, timestamp = bucket start. Rationale: V2's
    tool compares each series (kpi, node_id, service_name) in ±600 s against its own
    mean/std over ±1200 s; a single point can pass 3σ only when the baseline holds
    n >= 11 points ((n-1)/sqrt(n) bound). On 1-minute AIOps data the ±1200 s baseline
    holds ~41 points; an RCABench case spans ~8 min, so 10 s buckets give ~48 points
    (1-minute buckets: ~9, never flagged).
      container rows: rows with a pod; node_id = k8s node (Nezha: "node-1" placeholder),
                      service_name = pod (Nezha: pod name)
      node rows:      rows with a node and no pod; service_name = ""
      service rows:   rows with only a service (deployment / hubble KPIs); node_id = ""
                      (Nezha: front_service.csv rows)
      dropped:        rows without pod, node and service (cluster scope)
    No span-derived rr/sr/mrt/count rows: Nezha has none (V2 reads its given metric files).
metric/node_service_map.pkl / service_node_map.pkl
    From the container rows: {node: {pod}} and {pod: node} (Nezha keys pods too).
log/all/logs.csv                  (log_id, timestamp, cmdb_id, log_name, value)
    Nezha format: log_id "rcabench_<i>", timestamp epoch s, cmdb_id = pod (else service),
    log_name = "log_" + service (Nezha: "log_" + container), value =
    "severity: <level lowercased, 'unknown' if empty>, message: <message>".
    preprocessing_logs.py's filter then drops INFO/DEBUG rows (``severity:\\s*(info|debug)``)
    and rows are sorted by timestamp (tool_server applies the same filter at load).
error_traces.txt
    ``RCLV2_TOP_N_TRACES`` = N > 0 (default 5): upstream's Nezha rule
    (``preprocess_nezha.generate_error_traces``): the N longest parent-less spans
    (``nlargest(min(N, len), "duration")``) whose timestamp lies in the fault window. The fault
    window is the ABNORMAL window of env.json (ABNORMAL_START..ABNORMAL_END, inclusive), since
    RCABench gives no inject time to the agent; without env.json (the harness sandbox holds only the
    telemetry tables) it is the abnormal_traces time range (first to last span start, ms). If that window holds no root span, the N longest
    roots overall are taken and ``fallback`` is recorded (Nezha's own fallback, roots above the
    95th duration percentile, only runs when no fault matched). ``RCLV2_TOP_N_TRACES=0``:
    upstream's AIOps rule (preprocessing_traces.py) over all roots: non-ok status_code or
    duration > DURATION_THRESHOLD_US (10000000 µs), status-first, deduplicated by span_id.
    Trace dedup (``RCLV2_DEDUP_TRACES``, default 1): of the selected roots whose whole call trees are
    identical (same subtree signature as step B), only the longest is kept; the case may then have
    fewer than N error traces (no refill).
    In both cases rows are deduplicated by span_id, sorted by timestamp, index reset and written
    with ``DataFrame.to_string()`` + "\\n", as upstream does.

Cache: ``$RCLV2_NATIVE_DIR`` or ``<RCA_EVENTS_DIR or tmp>/rclv2_native/<case_key>``; rebuilt
only when the fingerprint (converter version, case dir, top-N) changes; built in a temporary
sibling and renamed into place.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import duckdb
import pandas as pd

_AGENTS = Path(__file__).resolve().parents[1]
if str(_AGENTS) not in sys.path:
    sys.path.append(str(_AGENTS))
from _rca_shared import cases  # noqa: E402

VERSION = "rclv2-native-8"
BUCKET_S = 10
DURATION_THRESHOLD_US = 10_000_000                     # config.DURATION_THRESHOLD_US default
SPAN_COLUMNS = ["timestamp", "cmdb_id", "span_id", "trace_id", "duration", "type", "status_code",
                "operation_name", "parent_span", "repeat_count"]
METRIC_COLUMNS = ["timestamp", "node_id", "service_name", "kpi_name", "value"]
LOG_COLUMNS = ["log_id", "timestamp", "cmdb_id", "log_name", "value"]
INFO_DEBUG_PATTERN = r"severity:\s*(info|debug)"       # preprocessing_logs.py
DISPLAY = ("display.max_rows", None, "display.max_columns", None, "display.width", None,
           "display.max_colwidth", None)               # preprocessing_traces.py options


def top_n_traces() -> int:
    return int(os.environ.get("RCLV2_TOP_N_TRACES", "5"))      # 0 = upstream AIOps rule


def dedup_traces() -> bool:
    return os.environ.get("RCLV2_DEDUP_TRACES", "1") != "0"


def span_compaction() -> bool:
    return os.environ.get("RCLV2_SPAN_COMPACTION", "1") != "0"


def _drop_reattach(df: pd.DataFrame, drop: pd.Series) -> pd.DataFrame:
    """Remove the ``drop`` rows; their children move up to the nearest kept ancestor."""
    up = dict(zip(df.loc[drop, "span_id"], df.loc[drop, "parent_span"]))
    par = df["parent_span"].copy()
    for _ in range(10_000):                            # one hop per nested dropped level
        hop = par.isin(list(up))
        if not hop.any():
            break
        par[hop] = par[hop].map(up)
    return df.assign(parent_span=par)[~drop]


def subtree_signatures(df: pd.DataFrame) -> pd.Series:
    """Per row: hash of (service, span name, status, set of its children's signatures), bottom-up."""
    ids = set(df["span_id"])
    top = df["parent_span"].isna() | ~df["parent_span"].isin(ids)
    depth = pd.Series(0, index=df.index)
    level, d = set(df.loc[top, "span_id"]), 0
    while level and d < 10_000:
        d += 1
        m = df["parent_span"].isin(level) & ~top
        depth[m] = d
        level = set(df.loc[m, "span_id"])
    own = (df["service"].fillna("") + "\x1f" + df["operation_name"].fillna("") + "\x1f"
           + df["status_code"].fillna(""))
    sig: dict = {}
    child_sigs: dict = {}
    for lv in range(int(depth.max()) if len(df) else -1, -1, -1):
        m = (depth == lv).to_numpy()
        by_parent: dict = {}
        for o, sid, p in zip(own.to_numpy()[m], df["span_id"].to_numpy()[m], df["parent_span"].to_numpy()[m]):
            h = hashlib.sha1((o + "\x1e" + "\x1d".join(sorted(child_sigs.get(sid, ())))).encode()).hexdigest()
            sig[sid] = h
            if not pd.isna(p):
                by_parent.setdefault(p, set()).add(h)
        child_sigs = by_parent
    return df["span_id"].map(sig)


def compact_spans(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Steps A and B of the module docstring. ``df`` needs span_id, parent_span (NaN for roots), type,
    service, operation_name, status_code, duration; returns the kept rows (original order) with
    ``repeat_count``, ``subtree_sig`` and per-step row counts."""
    n0 = len(df)
    # A: drop non-root INTERNAL spans
    df = _drop_reattach(df, df["type"].astype(str).str.upper().eq("INTERNAL") & df["parent_span"].notna())
    n_a = len(df)
    # B: merge siblings with identical subtrees into the longest one
    df = df.assign(repeat_count=1, subtree_sig=subtree_signatures(df))
    ids = set(df["span_id"])
    child = df[df["parent_span"].isin(ids)].sort_values(["duration", "span_id"], ascending=[False, True])
    groups = child.groupby([child["parent_span"], child["subtree_sig"]], sort=False)["span_id"]
    rep = groups.cumcount().eq(0)
    df.loc[rep[rep].index, "repeat_count"] = groups.transform("size")[rep].astype(int)
    dropped = set(child.loc[~rep, "span_id"])
    frontier = dropped
    while frontier:                                    # subtrees of merged spans go too
        frontier = set(df.loc[df["parent_span"].isin(frontier), "span_id"]) - dropped
        dropped |= frontier
    df = df[~df["span_id"].isin(dropped)]              # a kept span's signature is unchanged by B
    return df, {"span_rows_raw": n0, "internal_dropped": n0 - n_a,
                "sibling_merged_dropped": n_a - len(df), "span_rows_kept": len(df)}


def tree_sizes(df: pd.DataFrame, roots) -> list[int]:
    """Spans reachable from each root through parent links (what coordinator.build_trace_graph walks)."""
    kids = df.groupby("parent_span")["span_id"].apply(list).to_dict()
    sizes = []
    for r in roots:
        seen, stack = set(), [r]
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            stack.extend(kids.get(s, []))
        sizes.append(len(seen))
    return sizes


def case_key(case_dir: Path) -> str:
    case_dir = Path(case_dir)
    return case_dir.parent.name if case_dir.name == "converted" else case_dir.name


def default_root(case_dir: Path) -> Path:
    if os.environ.get("RCLV2_NATIVE_DIR"):
        return Path(os.environ["RCLV2_NATIVE_DIR"])
    base = os.environ.get("RCA_EVENTS_DIR") or tempfile.gettempdir()
    return Path(base) / "rclv2_native" / case_key(case_dir)


def _fingerprint(case_dir: Path) -> str:
    fp = {"v": VERSION, "case": str(Path(case_dir).resolve()), "top_n": top_n_traces(),
          "compaction": span_compaction(), "dedup": dedup_traces()}
    if cases.kind(case_dir) == "rca100":                 # the abnormal window comes from the question
        fp.update(dataset="rca100", window=cases.incident_window(case_dir))
    env = cases._read_env(case_dir)
    if env.get("ABNORMAL_START") and env.get("ABNORMAL_END"):
        fp["window"] = (env["ABNORMAL_START"], env["ABNORMAL_END"])
    payload = json.dumps(fp, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def convert(case_dir: str | Path, out_root: str | Path | None = None, force: bool = False) -> dict:
    case_dir = Path(case_dir)
    root = Path(out_root) if out_root else default_root(case_dir)
    fp = _fingerprint(case_dir)
    marker = root / "conversion_report.json"
    if not force and marker.exists():
        rep = json.loads(marker.read_text())
        if rep.get("fingerprint") == fp:
            rep["cached"] = True
            return rep
    root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".rclv2_native_", dir=str(root.parent)))
    try:
        rep = _convert_into(case_dir, work, fp)
        if root.exists():
            shutil.rmtree(root)
        work.rename(root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    rep["root"] = str(root)
    rep["case_key"] = case_key(case_dir)
    marker.write_text(json.dumps(rep, indent=1, sort_keys=True))
    rep["cached"] = False
    return rep


def _union(b, plane: str, select: str, where: str = "TRUE") -> str:
    parts = [f"SELECT {select}, '{w}' AS win FROM {v} WHERE {where}"
             for w, v in ((w, b.view(plane, w)) for w in ("reference", "incident")) if v]
    return " UNION ALL ".join(parts)


def _is_error_status(value) -> bool:                   # preprocessing_traces.py
    ok_values = {"ok", "0", "200", 0, 200}
    if pd.isna(value):
        return False
    try:
        return str(value).strip().lower() not in ok_values
    except Exception:
        return False


def _convert_into(case_dir: Path, root: Path, fp: str) -> dict:
    con = duckdb.connect()
    b = cases._bind(case_dir, con)
    for d in ("trace/all", "metric/all", "log/all"):
        (root / d).mkdir(parents=True)

    # ── spans ──
    sql = _union(b, "traces", """
        CAST(floor(epoch(ts) * 1000) AS BIGINT) AS timestamp,
        coalesce(nullif(json_extract_string(attributes, '$."k8s.pod.name"'), ''), service, '') AS cmdb_id,
        span_id, trace_id,
        CAST(round(duration_ns / 1000.0) AS BIGINT) AS duration,
        coalesce(kind, '') AS type,
        CASE WHEN is_error THEN 'error' ELSE '0' END AS status_code,
        coalesce(span_name, '') AS operation_name,
        parent_id AS parent_span,
        coalesce(service, '') AS service""")
    con.execute(f"CREATE TEMP TABLE s AS SELECT * FROM ({sql}) ORDER BY timestamp, trace_id, span_id")
    raw = con.execute(f"SELECT {', '.join(SPAN_COLUMNS[:-1])}, service FROM s "
                      f"ORDER BY timestamp, trace_id, span_id").df()
    if span_compaction():
        spans, compaction = compact_spans(raw)
    else:
        spans, compaction = raw.assign(repeat_count=1), {"span_rows_raw": len(raw), "span_rows_kept": len(raw)}
    compaction["enabled"] = span_compaction()
    sigs = spans["subtree_sig"] if "subtree_sig" in spans else subtree_signatures(spans)
    root_sig = dict(zip(spans.loc[spans["parent_span"].isna(), "span_id"], sigs[spans["parent_span"].isna()]))
    spans = spans[SPAN_COLUMNS]
    span_csv = root / "trace" / "all" / "trace_jaeger-span.csv"
    spans.to_csv(span_csv, index=False)

    # ── metrics (BUCKET_S means) ──
    con.execute("CREATE TEMP TABLE m0 AS " + _union(
        b, "metrics", "epoch(ts) AS t, metric, value, service, pod, node",
        "value IS NOT NULL AND NOT isnan(value) AND metric IS NOT NULL"))
    raw_rows = con.execute("SELECT count(*) FROM m0").fetchone()[0]
    dropped = con.execute("SELECT count(*) FROM m0 WHERE pod IS NULL AND node IS NULL AND service IS NULL").fetchone()[0]
    bucket = f"CAST(floor(t / {BUCKET_S}) * {BUCKET_S} AS BIGINT)"
    blocks = {
        "container": f"SELECT {bucket}, coalesce(node, ''), pod, metric, avg(value) FROM m0 "
                     f"WHERE pod IS NOT NULL GROUP BY 1, 2, 3, 4",
        "node": f"SELECT {bucket}, node, '', metric, avg(value) FROM m0 "
                f"WHERE pod IS NULL AND node IS NOT NULL GROUP BY 1, 2, 3, 4",
        "service": f"SELECT {bucket}, '', service, metric, avg(value) FROM m0 "
                   f"WHERE pod IS NULL AND node IS NULL AND service IS NOT NULL GROUP BY 1, 2, 3, 4",
    }
    frames, counts = [], {}
    for name, q in blocks.items():
        df = con.execute(f"SELECT * FROM ({q}) ORDER BY 3, 4, 2, 1").df()
        df.columns = METRIC_COLUMNS
        counts[name] = len(df)
        frames.append(df)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(root / "metric" / "all" / "metrics.csv", index=False)
    node_service_map: dict = {}
    service_node_map: dict = {}
    cont = frames[0][["node_id", "service_name"]].drop_duplicates().sort_values(["service_name", "node_id"])
    for node_id, pod in zip(cont.node_id, cont.service_name):
        if node_id:
            node_service_map.setdefault(node_id, set()).add(pod)
            service_node_map[pod] = node_id
    with open(root / "metric" / "node_service_map.pkl", "wb") as f:
        pickle.dump(node_service_map, f)
    with open(root / "metric" / "service_node_map.pkl", "wb") as f:
        pickle.dump(service_node_map, f)

    # ── logs (Nezha format, preprocessing_logs.py filter) ──
    lsql = _union(b, "logs", """CAST(floor(epoch(ts)) AS BIGINT) AS timestamp,
        coalesce(nullif(pod, ''), service, '') AS cmdb_id,
        'log_' || coalesce(service, '') AS log_name,
        'severity: ' || coalesce(nullif(lower(level), ''), 'unknown') || ', message: ' || coalesce(message, '') AS value""")
    logs = con.execute(f"SELECT timestamp, cmdb_id, log_name, value FROM ({lsql}) "
                       f"ORDER BY timestamp, cmdb_id, value").df() if lsql else pd.DataFrame(columns=LOG_COLUMNS[1:])
    logs.insert(0, "log_id", [f"rcabench_{i}" for i in range(len(logs))])
    n_logs = len(logs)
    with warnings.catch_warnings():                    # upstream pattern has a capture group
        warnings.simplefilter("ignore", UserWarning)
        noise = logs["value"].astype(str).str.lower().str.contains(INFO_DEBUG_PATTERN, regex=True, na=False)
    logs = logs[~noise].sort_values("timestamp", kind="stable").reset_index(drop=True)
    logs.to_csv(root / "log" / "all" / "logs.csv", index=False)
    con.close()

    # ── error traces ──
    df = pd.read_csv(span_csv, dtype={"status_code": str})  # as tool_server.py reads it
    root_spans = df[df["parent_span"].isna()].copy()
    # abnormal window: env.json when present, else the abnormal_traces time range (the harness
    # sandbox holds only the telemetry tables, no env.json)
    env = cases._read_env(case_dir)
    if env.get("ABNORMAL_START") and env.get("ABNORMAL_END"):
        window_source = "question alert window" if cases.kind(case_dir) == "rca100" else "env.json"
        a0, a1 = int(env["ABNORMAL_START"]) * 1000, int(env["ABNORMAL_END"]) * 1000
    else:
        window_source = "abnormal_traces range"
        a0, a1 = (int(v) for v in duckdb.sql(
            f"SELECT min(epoch_ms(time)), max(epoch_ms(time)) "
            f"FROM read_parquet('{case_dir / 'abnormal_traces.parquet'}')").fetchone())
    window = root_spans[root_spans["timestamp"].between(a0, a1)]
    top_n, fallback = top_n_traces(), False
    if top_n > 0:                                      # preprocess_nezha.generate_error_traces
        if len(window) > 0:
            selected = [window.nlargest(min(top_n, len(window)), "duration")]
        else:
            fallback = True
            selected = [root_spans.nlargest(min(top_n, len(root_spans)), "duration")]
        rule = f"nezha_top{top_n}_abnormal_window"
    else:                                              # preprocessing_traces.find_error_traces
        selected = [root_spans[root_spans["status_code"].apply(_is_error_status)],
                    root_spans[root_spans["duration"] > DURATION_THRESHOLD_US]]
        rule = "aiops_status_or_duration"
    combined = pd.concat(selected).drop_duplicates(subset=["span_id"])
    n_selected, duplicates = len(combined), []
    if dedup_traces():                                 # D: one trace per distinct call tree
        combined = combined.sort_values(["duration", "span_id"], ascending=[False, True])
        dup = combined["span_id"].map(root_sig).duplicated()
        duplicates = combined.loc[dup, "span_id"].tolist()
        combined = combined[~dup]
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    with pd.option_context(*DISPLAY):
        (root / "error_traces.txt").write_text(combined.to_string() + "\n")

    # trace-tree sizes of the error traces (cost estimate: one Dedicated Agent per span)
    sizes = tree_sizes(df, combined.span_id)
    raw_sizes = tree_sizes(raw, combined.span_id)
    del raw
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return {
        "fingerprint": fp, "version": VERSION, "case_dir": str(case_dir), "bucket_s": BUCKET_S,
        "span_rows": len(spans), "span_compaction": compaction, "root_spans": int(df["parent_span"].isna().sum()),
        "incident_root_spans": int(len(window)),
        "metric_rows": len(metrics), "metric_rows_by_block": counts, "raw_metric_rows": int(raw_rows),
        "metric_rows_dropped_no_entity": int(dropped),
        "log_rows_all": int(n_logs), "log_rows_kept": int(len(logs)),
        "pods_in_map": len(service_node_map), "nodes_in_map": len(node_service_map),
        "error_traces": {"rule": rule, "top_n": top_n, "window_ms": [a0, a1], "window_source": window_source, "window_roots": int(len(window)),
                         "fallback": fallback, "selected": int(n_selected), "dedup": dedup_traces(),
                         "duplicate_trees_dropped": duplicates, "kept": int(len(combined)), "tree_sizes": sizes,
                         "tree_spans_total": int(sum(sizes)), "tree_sizes_uncompacted": raw_sizes,
                         "rows": combined.to_dict(orient="records")},
        "bytes": size,
    }


def summary(rep: dict) -> str:
    e = rep["error_traces"]
    return (f"[rclv2-native] {rep.get('root')} ({'cached' if rep.get('cached') else 'built'}): "
            f"{rep['span_rows']} spans ({rep['incident_root_spans']} incident roots), "
            f"{rep['metric_rows']} metric rows @{rep['bucket_s']}s {rep['metric_rows_by_block']}, "
            f"logs {rep['log_rows_kept']}/{rep['log_rows_all']} kept, maps {rep['pods_in_map']} pods / "
            f"{rep['nodes_in_map']} nodes, error traces {e['rule']} kept={e['kept']}/{e.get('selected')} fallback={e['fallback']} "
            f"spans/tree={e['tree_sizes']} (uncompacted {e.get('tree_sizes_uncompacted')}), {rep['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--question", default="", help="harness question (RCA100: source of the alert window)")
    a = ap.parse_args()
    cases.use(a.case_dir)
    cases.set_question(a.question)
    print(summary(convert(a.case_dir, a.out, a.force)))

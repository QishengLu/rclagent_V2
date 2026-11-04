#!/usr/bin/env python
"""Estimate RCLAgent-V2 cost on RCABench cases without any LLM call or conversion.

Applies the same trace selection as native_data.py (the RCLV2_TOP_N_TRACES longest parent-less root
spans in the abnormal window; Nezha rule) directly on the parquet, measures each selected trace's
tree size (spans reachable from the root through parent links — what coordinator.build_trace_graph
walks) after native_data's span compaction (RCLV2_SPAN_COMPACTION, default on; the uncompacted size of
all N selected trees is printed too) and trace dedup (RCLV2_DEDUP_TRACES, default on), and estimates
LLM calls:

    per trace  ≈ spans × (verification rounds + 1 consolidation)      (default 2 + 1 = 3 per span)
    per case   ≈ Σ per trace + 1 case-level final call

Usage:
  agents/_rca_shared/.venv/bin/python agents/rclagent_v2/estimate_cost.py \
      --data-root datasets/RCABench-500 --cases-file cases100.txt \
      [--top-n 5] [--calls-per-span 3] [--sec-per-call 4] [--concurrency 16] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_data  # noqa: E402


def case_stats(converted: Path, top_n: int) -> dict:
    con = duckdb.connect()
    tables = [str(converted / f"{w}_traces.parquet") for w in ("normal", "abnormal")
              if (converted / f"{w}_traces.parquet").exists()]
    con.execute(f"""CREATE VIEW t AS SELECT span_id, nullif(parent_span_id, '') AS parent_span, trace_id,
        duration, upper(coalesce("attr.span_kind", '')) AS "type", coalesce(service_name, '') AS service,
        coalesce(span_name, '') AS operation_name,
        CASE WHEN "attr.status_code" = 'Error' THEN 'error' ELSE '0' END AS status_code
        FROM read_parquet({tables!r})""")
    ab = converted / "abnormal_traces.parquet"
    roots = con.execute(f"""SELECT span_id, trace_id, duration FROM read_parquet('{ab}')
                            WHERE parent_span_id IS NULL OR parent_span_id = ''
                            ORDER BY duration DESC LIMIT {int(top_n)}""").fetchall()
    n_roots = con.execute(f"SELECT count(*) FROM read_parquet('{ab}') "
                          "WHERE parent_span_id IS NULL OR parent_span_id = ''").fetchone()[0]
    sizes, raw_sizes, seen, dups = [], [], set(), 0
    for span_id, trace_id, _ in roots:                 # longest first, as native_data's dedup keeps
        df = con.execute("SELECT * FROM t WHERE trace_id = ?", [trace_id]).df()
        raw_sizes += native_data.tree_sizes(df, [span_id])
        if native_data.span_compaction():
            df = native_data.compact_spans(df)[0]
        if native_data.dedup_traces():
            sig = dict(zip(df.span_id, df["subtree_sig"] if "subtree_sig" in df
                           else native_data.subtree_signatures(df)))[span_id]
            if sig in seen:
                dups += 1
                continue
            seen.add(sig)
        sizes += native_data.tree_sizes(df, [span_id])
    con.close()
    return {"incident_roots": n_roots, "tree_sizes": sizes, "spans": sum(sizes), "raw_sizes": raw_sizes,
            "duplicate_traces": dups}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="dir holding <case>/converted/")
    ap.add_argument("--cases-file", required=True, help="one case name per line")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--calls-per-span", type=float, default=3.0)
    ap.add_argument("--sec-per-call", type=float, default=4.0)
    ap.add_argument("--concurrency", type=int, default=16, help="total concurrent LLM calls")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    cases = [c.strip() for c in Path(a.cases_file).read_text().splitlines() if c.strip()]
    rows = []
    for c in cases:
        d = Path(a.data_root) / c / "converted"
        try:
            s = case_stats(d, a.top_n)
        except Exception as e:  # noqa: BLE001
            print(f"{c}: ERROR {type(e).__name__}: {e}")
            continue
        calls = round(s["spans"] * a.calls_per_span + 1)
        rows.append({"case": c, "incident_roots": s["incident_roots"], "traces": len(s["tree_sizes"]),
                     "tree_sizes": " ".join(map(str, s["tree_sizes"])), "spans": s["spans"],
                     "spans_uncompacted": sum(s["raw_sizes"]), "duplicate_traces": s["duplicate_traces"],
                     "est_calls": calls})
        print(f"{c}: traces={len(s['tree_sizes'])} spans/tree={s['tree_sizes']} total_spans={s['spans']} "
              f"(uncompacted {sum(s['raw_sizes'])}) "
              f"est_calls={calls}")
    if not rows:
        return
    sp = [r["spans"] for r in rows]
    ca = [r["est_calls"] for r in rows]
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    total = sum(ca)
    print(f"\n{len(rows)} cases | spans per case: median {st.median(sp):.0f}, p90 {q(sp, .9)}, max {max(sp)}"
          f" | calls per case: median {st.median(ca):.0f}, p90 {q(ca, .9)}, max {max(ca)}")
    print(f"total est. calls: {total:,} "
          f"(≈ {total * a.sec_per_call / a.concurrency / 3600:.1f} h at {a.concurrency} concurrent × "
          f"{a.sec_per_call:g} s/call)")
    print(f"heaviest case: {max(rows, key=lambda r: r['spans'])['case']} "
          f"(≈ {max(ca) * a.sec_per_call / 8 / 3600:.1f} h at 8 parallel calls within the case)")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

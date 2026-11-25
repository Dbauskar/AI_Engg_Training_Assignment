"""Per-stage latency benchmark: 200 queries after warm-up, p50/p95/p99 vs the design budgets.

Usage: uv run python scripts/bench_rag.py [--tenant acme-corp] [--n 200] [--threads N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.rag_queries import ALL  # noqa: E402
from mcp_servers.rag_pipeline import HybridRetriever  # noqa: E402

BUDGET_P95 = {"bm25": 12, "dense": 25, "rrf": 3, "rerank": 45}  # design.md section 4.1 (KPI-07/08/09)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="acme-corp")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    r = HybridRetriever(threads=a.threads)
    for q, _ in ALL:  # warm-up (session init, allocator, caches)
        r.retrieve(a.tenant, q)
    rows: dict[str, list[float]] = {k: [] for k in BUDGET_P95}
    pool_sizes = []
    for i in range(a.n):
        out = r.retrieve(a.tenant, ALL[i % len(ALL)][0])
        for k, v in out.stage_latency_ms.items():
            rows[k].append(v)
    print(f"{a.n} queries, tenant={a.tenant}, chunks={len(r.index_for(a.tenant).chunks)}, threads={a.threads or 'all'}")
    print(f"{'stage':8s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'budget p95':>11s}  result")
    for k, v in rows.items():
        p50, p95, p99 = np.percentile(v, [50, 95, 99])
        print(f"{k:8s} {p50:8.1f} {p95:8.1f} {p99:8.1f} {BUDGET_P95[k]:11d}  {'PASS' if p95 <= BUDGET_P95[k] else 'FAIL'}")


if __name__ == "__main__":
    main()

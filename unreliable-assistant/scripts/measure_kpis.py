"""Measure the cheap, isolated KPIs on the local stack and write evals/results/kpi_extra.json.
  uv run python scripts/measure_kpis.py [--n 200]
KPI-01 hook.pre p95 (real plugin chain), KPI-02 PII recall (held-out TEST), KPI-04/05 triage accuracy + p95 (held-out TEST),
KPI-07/08/09 retrieval stage p95. Everything else is measured elsewhere or reported NOT MEASURED. Sequential, one request at a time:
these are uncontended latencies, NOT latencies under load (see load_k6.json for that)."""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.triage import TriageAgent  # noqa: E402
from agents.triage_examples import ROUTE_OF  # noqa: E402
from app.localstack import build_local_runtime  # noqa: E402
from app.pii import PiiDetector  # noqa: E402
from app.runtime import Settings  # noqa: E402
from evals.pii_set import TEST as PII_TEST  # noqa: E402
from evals.rag_queries import ALL  # noqa: E402
from evals.triage_set import TEST as TRIAGE_TEST  # noqa: E402
from mcp_servers.rag_pipeline import HybridRetriever  # noqa: E402
from plugins.registry import RequestContext  # noqa: E402

SCALE = "local in-memory stack, sequential (uncontended)"


def p95(x):
    return float(np.percentile(x, 95))


async def main(n: int) -> dict:
    out: dict = {}
    s = Settings.from_env()
    rt = await build_local_runtime(s)
    await rt.warmup()
    lat = []
    for i in range(n):
        ctx = RequestContext.create("acme-corp", "u1", "s1", f"{ALL[i % len(ALL)][0]} (case {i})")  # unique text: no embedding-memo hits
        t = time.perf_counter()
        await rt.registry.run_pre(ctx)
        lat.append((time.perf_counter() - t) * 1000)
    out["KPI-01"] = {"value": p95(lat[10:]), "scale": SCALE}
    await rt.close()

    det = PiiDetector(use_ner=True)
    det.redact("warm up John Smith")
    total = leaks = 0
    for text, secrets in PII_TEST:
        r = det.redact(text)
        for sec in secrets:
            total += 1
            leaks += sec in r.text
    out["KPI-02"] = {"value": (total - leaks) / total, "scale": f"{SCALE}; {total} held-out PII items"}

    retr = HybridRetriever()
    agent = TriageAgent(retr.embedder)
    for q, _ in TRIAGE_TEST[:5]:
        agent.classify(q)
    ok, tl = 0, []
    for q, label in TRIAGE_TEST:
        r = agent.classify(q)
        tl.append(r.latency_ms)
        ok += r.route == ROUTE_OF[label] and (ROUTE_OF[label] != "HYBRID_RAG" or r.domain == label)
    out["KPI-04"] = {"value": ok / len(TRIAGE_TEST), "scale": f"{SCALE}; {len(TRIAGE_TEST)} held-out queries"}
    out["KPI-05"] = {"value": p95(tl), "scale": SCALE}

    for q, _ in ALL:
        retr.retrieve("acme-corp", q)
    rows: dict[str, list[float]] = {"bm25": [], "dense": [], "rerank": []}
    for i in range(n):
        o = retr.retrieve("acme-corp", ALL[i % len(ALL)][0])
        for k in rows:
            rows[k].append(o.stage_latency_ms[k])
    for kid, k in (("KPI-07", "bm25"), ("KPI-08", "dense"), ("KPI-09", "rerank")):
        out[kid] = {"value": p95(rows[k]), "scale": f"{SCALE}; 12-chunk index (latency would grow with a real corpus)"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    res = asyncio.run(main(a.n))
    p = ROOT / "evals" / "results" / "kpi_extra.json"
    p.write_text(json.dumps({"kpis": res}, indent=2))
    for k, v in sorted(res.items()):
        print(f"{k}: {v['value']:.4g}")

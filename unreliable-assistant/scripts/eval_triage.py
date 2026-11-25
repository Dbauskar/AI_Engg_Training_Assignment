"""Evaluate triage on the held-out sets: accuracy per route, false refusals, latency. Tunes min_sim on DEV only.

Usage: uv run python scripts/eval_triage.py [--final]   (--final also reports TEST at the DEV-chosen threshold)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.triage import TriageAgent  # noqa: E402
from agents.triage_examples import ROUTE_OF  # noqa: E402
from evals.triage_set import DEV, TEST  # noqa: E402
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder  # noqa: E402


def score(agent, data):
    ok, false_refusals, in_scope, lat, wrong = 0, 0, 0, [], []
    for q, label in data:
        r = agent.classify(q)
        lat.append(r.latency_ms)
        good = r.route == ROUTE_OF[label] and (ROUTE_OF[label] != "HYBRID_RAG" or r.domain == label)
        ok += good
        if not good:
            wrong.append((q, label, r.route, r.domain, round(r.confidence, 2)))
        if ROUTE_OF[label] != "REJECT_OOS":
            in_scope += 1
            false_refusals += r.route == "REJECT_OOS"
    return ok / len(data), false_refusals / in_scope, lat, wrong


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--embed-dir", default=DEFAULT_EMBED_DIR)
    a = ap.parse_args()
    emb = OnnxEmbedder(a.embed_dir, threads=2)
    best = None
    for t in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        acc, fr, lat, _ = score(TriageAgent(emb, min_sim=t), DEV)
        print(f"DEV min_sim={t:.2f} accuracy={acc:.3f} false_refusal={fr:.3f}")
        if best is None or (acc, -fr) > (best[1], -best[2]):
            best = (t, acc, fr)
    print("chosen on DEV:", best[0])
    if a.final:
        agent = TriageAgent(emb, min_sim=best[0])
        for q, _ in TEST[:5]:
            agent.classify(q)  # warm-up
        acc, fr, lat, wrong = score(agent, TEST)
        p50, p95 = np.percentile(lat, [50, 95])
        print(f"TEST accuracy={acc:.3f} (KPI-04 >= 0.97)  false_refusal={fr:.3f} (KPI-06 <= 0.01)  latency p50={p50:.1f} p95={p95:.1f} ms (KPI-05 <= 25)")
        for w in wrong:
            print("  WRONG:", w)

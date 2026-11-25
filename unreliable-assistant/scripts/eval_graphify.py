"""Token reduction + answer coverage of Graphify context over the 100 evaluation queries.

Baselines (the 'before'):
  naive   the top-4 reranked chunks, full text, regardless of score (a plain RAG prompt)
  actual  the chunks the pipeline really keeps (score > 0.42) rendered uncompressed
  today   the 180-token compressed observation the ReAct loop already builds (agents/react.py)
'After' = the Graphify subgraph (<= 320 token budget), restricted to the policies retrieval returned.
Usage: uv run python scripts/eval_graphify.py [--budget 320]
"""
import argparse, sys, tempfile
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.react import compress_observation  # noqa: E402
from app.tokens import count_tokens  # noqa: E402
from evals.graphify_queries import QUERIES  # noqa: E402
from mcp_servers.graphify import Graph, build_graph  # noqa: E402
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, HybridRetriever, build_index  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=int, default=320); ap.add_argument("--dense", action="store_true"); ap.add_argument("--anchors", type=int, default=1); ap.add_argument("--min-rel", type=float, default=0.25); ap.add_argument("--json"); a = ap.parse_args()
root = Path(tempfile.mkdtemp()); r = HybridRetriever(root)
g = Graph(build_graph(ROOT), r.embedder if a.dense else None)
tok = Tokenizer.from_file(f"{DEFAULT_EMBED_DIR}/tokenizer.json"); tok.no_truncation(); tok.no_padding()
build_index("acme-corp", ROOT / "knowledge_vault", root, r.embedder, tok)
r.retrieve("acme-corp", "warm up")
rows = []
for q, gold in QUERIES:
    naive = r.retrieve("acme-corp", q, score_threshold=0.0, rerank_k=4)
    actual = r.retrieve("acme-corp", q)
    hubs = sorted({c.chunk_id.split("-")[0] for c in naive.chunks})
    t_naive = count_tokens("\n".join(c.text for c in naive.chunks))
    t_actual = count_tokens("\n".join(c.text for c in actual.chunks))
    t_today = count_tokens(compress_observation(actual.model_dump(), set(), 180)) if actual.chunks else 0
    for mode, hub_ids in (("restricted", hubs), ("free", None)):
        sg = g.subgraph(q, 2, a.budget, hub_ids, a.min_rel, max_anchors=a.anchors)
        ids = {n.id.split(":", 1)[1] for n in sg.nodes}
        rows.append(dict(mode=mode, q=q, gold=gold, anchor_ok=bool(sg.anchor) and sg.anchor.split(":", 1)[1] in gold, covered=all(x in ids for x in gold),
                         tn=t_naive, ta=t_actual, tt=t_today, tg=sg.tokens, nodes=len(sg.nodes)))
def pct(x): return f"{x:6.1%}"
print(f"{len(QUERIES)} queries | graph: {len(g.nodes)} nodes | budget {a.budget} tokens\n")
for mode in ("restricted", "free"):
    R = [x for x in rows if x["mode"] == mode]
    anchor, cov = np.mean([x["anchor_ok"] for x in R]), np.mean([x["covered"] for x in R])
    print(f"[{mode}] anchor is a gold note: {pct(anchor)} | ALL gold notes in subgraph: {pct(cov)} | mean nodes {np.mean([x['nodes'] for x in R]):.1f}")
    for name, key in (("naive top-4 chunks", "tn"), ("actual kept chunks", "ta"), ("today's 180-tok observation", "tt")):
        base = np.array([x[key] for x in R], float); aft = np.array([x["tg"] for x in R], float)
        ok = base > 0
        per = 1 - aft[ok] / base[ok]
        print(f"   vs {name:28s}: before {base.mean():6.1f} -> after {aft.mean():5.1f} tokens | mean per-query reduction {per.mean():6.1%} | reduction of the means {1 - aft.mean() / base.mean():6.1%} | queries where graph is LARGER: {(aft[ok] > base[ok]).sum()}")
    miss = [x for x in R if not x["covered"]]
    if mode == "restricted":
        for x in miss[:8]: print("      NOT COVERED:", x["q"][:70], "| gold:", x["gold"])

if a.json:
    import json
    R = [x for x in rows if x["mode"] == "restricted"]
    out = {"queries": len(R), "budget": a.budget, "dense": a.dense, "anchors": a.anchors, "graph_nodes": len(g.nodes),
           "anchor_is_gold": float(np.mean([x["anchor_ok"] for x in R])), "all_gold_covered": float(np.mean([x["covered"] for x in R])),
           "mean_tokens": {"naive_top4": float(np.mean([x["tn"] for x in R])), "actual_kept": float(np.mean([x["ta"] for x in R])),
                           "today_compressed": float(np.mean([x["tt"] for x in R])), "graph": float(np.mean([x["tg"] for x in R]))},
           "reduction_vs_naive": float(1 - np.mean([x["tg"] for x in R]) / np.mean([x["tn"] for x in R])),
           "reduction_vs_actual": float(1 - np.mean([x["tg"] for x in R]) / np.mean([x["ta"] for x in R])),
           "note": "KPI-26 target >= 45% is met only against the naive baseline on this toy corpus (chunks ~50 tokens)."}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(out, indent=2)); print("wrote", a.json)

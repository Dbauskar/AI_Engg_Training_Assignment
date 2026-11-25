"""Build graph.json for a tenant from the vault + project code.
Usage: uv run python scripts/graphify_build.py [--tenant acme-corp] [--out data/graph]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp_servers.graphify import build_graph, save_graph  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tenant", default="acme-corp"); ap.add_argument("--out", default="data/graph"); a = ap.parse_args()
    g = build_graph(ROOT)
    out = ROOT / a.out / a.tenant / "graph.json"
    save_graph(g, out)
    types = {}
    for n in g["nodes"]:
        types[n["type"]] = types.get(n["type"], 0) + 1
    etypes = {}
    for e in g["edges"]:
        etypes[e["type"]] = etypes.get(e["type"], 0) + 1
    print(f"wrote {out.relative_to(ROOT)}: nodes {types} edges {etypes} dangling {len(g['dangling'])}")
    for d in g["dangling"]:
        print("  DANGLING LINK:", d)

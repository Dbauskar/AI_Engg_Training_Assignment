"""Build a per-tenant hybrid index from Markdown notes.

Usage: uv run python scripts/ingest.py --tenant acme-corp [--src knowledge_vault] [--out data/index]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make project packages importable

from tokenizers import Tokenizer  # noqa: E402

from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder, build_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--src", default="knowledge_vault")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--embed-dir", default=DEFAULT_EMBED_DIR)
    a = ap.parse_args()
    t0 = time.perf_counter()
    emb = OnnxEmbedder(a.embed_dir)
    chunk_tok = Tokenizer.from_file(str(Path(a.embed_dir) / "tokenizer.json"))
    chunk_tok.no_truncation()
    chunk_tok.no_padding()
    out = build_index(a.tenant, a.src, a.out, emb, chunk_tok)
    print(f"indexed tenant={a.tenant} -> {out} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

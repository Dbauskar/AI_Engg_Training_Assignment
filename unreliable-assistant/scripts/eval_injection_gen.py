"""Evaluate the injection detector (patterns + classifier) on evals/injection_gen/holdout3.json. Prints rates only."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.injection import EmbeddingClassifier, InjectionDetector, LexicalClassifier  # noqa: E402
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder  # noqa: E402

H = json.loads((ROOT / "evals/injection_gen/holdout3.json").read_text())
mode = sys.argv[1] if len(sys.argv) > 1 else "both"   # lex | emb | both
emb = EmbeddingClassifier(OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2)) if mode in ("emb", "both") else None
det = InjectionDetector(classifier=emb, lexical=LexicalClassifier() if mode in ("lex", "both") else None)
tot = hit = 0
for k in ("direct", "indirect"):
    b = sum(det.scan(t).is_attack for t in H[k]); tot += len(H[k]); hit += b
    print(f"{k}: blocked {b}/{len(H[k])} = {b / len(H[k]):.1%}")
fp = sum(det.scan(t).is_attack for t in H["benign"])
print(f"overall block {hit}/{tot} = {hit / tot:.1%} | benign FP {fp}/{len(H['benign'])} = {fp / len(H['benign']):.1%}")

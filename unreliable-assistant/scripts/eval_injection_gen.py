"""Evaluate the injection detector (patterns + classifier) on evals/injection_gen/holdout3.json. Prints rates only."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.injection import EmbeddingClassifier, InjectionDetector, LexicalClassifier  # noqa: E402
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder  # noqa: E402

H = json.loads((ROOT / "evals/injection_gen/holdout3.json").read_text())
mode = sys.argv[1] if len(sys.argv) > 1 else "both"   # lex | emb | both | esc (both + the LLM escalation judge)
emb = EmbeddingClassifier(OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2)) if mode in ("emb", "both", "esc") else None
esc = None
if mode == "esc":
    import os
    from agents.llm_verifier import LLMInjectionJudge
    from app.config import load_env
    load_env()
    esc = LLMInjectionJudge(os.environ["OPENROUTER_API_KEY"])
det = InjectionDetector(classifier=emb, lexical=LexicalClassifier() if mode in ("lex", "both", "esc") else None, escalator=esc)
ESC = mode == "esc"
tot = hit = 0
for k in ("direct", "indirect"):
    b = sum(det.scan(t, escalate=ESC).is_attack for t in H[k]); tot += len(H[k]); hit += b
    print(f"{k}: blocked {b}/{len(H[k])} = {b / len(H[k]):.1%}")
fp = sum(det.scan(t, escalate=ESC).is_attack for t in H["benign"])
print(f"overall block {hit}/{tot} = {hit / tot:.1%} | benign FP {fp}/{len(H['benign'])} = {fp / len(H['benign']):.1%}")

if ESC:
    from evals import injection_holdout as H2
    atk = H2.DIRECT + H2.INDIRECT
    b = sum(det.scan(t, escalate=True).is_attack for t in atk); fp2 = sum(det.scan(t, escalate=True).is_attack for t in H2.BENIGN)
    print(f"original held-out set: blocked {b}/{len(atk)} = {b / len(atk):.1%} | benign FP {fp2}/{len(H2.BENIGN)}")

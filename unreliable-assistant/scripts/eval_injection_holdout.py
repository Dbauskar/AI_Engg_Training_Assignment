"""Held-out injection check (evals/injection_holdout.py). Prints rates and ids only, never payloads.
Usage: uv run python scripts/eval_injection_holdout.py [--semantic]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.injection import InjectionDetector  # noqa: E402
from evals import injection_holdout as H  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--semantic", action="store_true"); ap.add_argument("--clf", action="store_true")
ap.add_argument("--esc", action="store_true", help="patterns + classifiers + the LLM escalation judge, pooled over BOTH held-out sets (needs OPENROUTER_API_KEY)")
a = ap.parse_args()
if a.esc:
    a.clf = True
if a.clf:
    from app.injection import EmbeddingClassifier
    from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder
    e = OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2)
    from app.injection import LexicalClassifier
    esc = None
    if a.esc:
        import os
        from agents.llm_verifier import LLMInjectionJudge
        from app.config import load_env
        load_env()
        esc = LLMInjectionJudge(os.environ["OPENROUTER_API_KEY"])
    det = InjectionDetector(classifier=EmbeddingClassifier(e), lexical=LexicalClassifier(), escalator=esc)
elif a.semantic:
    from app.injection_examples import ATTACK, BENIGN
    from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder
    det = InjectionDetector(OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2), ATTACK, BENIGN)
else:
    det = InjectionDetector()
res = {}
SETS = {"direct": list(H.DIRECT), "indirect": list(H.INDIRECT)}
BENIGN = list(H.BENIGN)
if a.esc:  # pool the human-written held-out set with the LLM-generated one that training never read
    import json as _j
    g = _j.loads((Path(__file__).resolve().parents[1] / "evals/injection_gen/holdout3.json").read_text())
    SETS["direct"] += g["direct"]; SETS["indirect"] += g["indirect"]; BENIGN += g["benign"]
for name, items in SETS.items():
    miss = [i for i, t in enumerate(items) if not det.scan(t, escalate=a.esc).is_attack]
    res[name] = {"n": len(items), "blocked": len(items) - len(miss), "missed_ids": miss}
    print(f"{name}: blocked {len(items) - len(miss)}/{len(items)} = {(len(items) - len(miss)) / len(items):.1%} missed ids {miss}")
fp = [i for i, t in enumerate(BENIGN) if det.scan(t, escalate=a.esc).is_attack]
print(f"benign: false positives {len(fp)}/{len(BENIGN)} = {len(fp) / len(BENIGN):.1%} ids {fp}")
res["benign"] = {"n": len(BENIGN), "false_positives": len(fp), "ids": fp}
res["mode"] = "patterns+classifiers+llm-escalation, pooled: 40 human-written + 244 LLM-generated held-out attacks" if a.esc else "patterns+classifier" if a.clf else ("patterns+nearest-neighbour" if a.semantic else "patterns only")
out = Path(__file__).resolve().parents[1] / "evals" / "results" / "injection_holdout.json"
out.write_text(json.dumps(res, indent=2))

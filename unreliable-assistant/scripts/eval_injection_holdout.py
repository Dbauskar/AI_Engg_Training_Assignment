"""Held-out injection check (evals/injection_holdout.py). Prints rates and ids only, never payloads.
Usage: uv run python scripts/eval_injection_holdout.py [--semantic]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.injection import InjectionDetector  # noqa: E402
from evals import injection_holdout as H  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--semantic", action="store_true"); ap.add_argument("--clf", action="store_true"); a = ap.parse_args()
if a.clf:
    from app.injection import EmbeddingClassifier
    from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder
    e = OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2)
    from app.injection import LexicalClassifier
    det = InjectionDetector(classifier=EmbeddingClassifier(e), lexical=LexicalClassifier())
elif a.semantic:
    from app.injection_examples import ATTACK, BENIGN
    from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder
    det = InjectionDetector(OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2), ATTACK, BENIGN)
else:
    det = InjectionDetector()
res = {}
for name, items in (("direct", H.DIRECT), ("indirect", H.INDIRECT)):
    miss = [i for i, t in enumerate(items) if not det.scan(t).is_attack]
    res[name] = {"n": len(items), "blocked": len(items) - len(miss), "missed_ids": miss}
    print(f"{name}: blocked {len(items) - len(miss)}/{len(items)} = {(len(items) - len(miss)) / len(items):.1%} missed ids {miss}")
fp = [i for i, t in enumerate(H.BENIGN) if det.scan(t).is_attack]
print(f"benign: false positives {len(fp)}/{len(H.BENIGN)} = {len(fp) / len(H.BENIGN):.1%} ids {fp}")
res["benign"] = {"n": len(H.BENIGN), "false_positives": len(fp), "ids": fp}
res["mode"] = "patterns+classifier" if a.clf else ("patterns+nearest-neighbour" if a.semantic else "patterns only")
out = Path(__file__).resolve().parents[1] / "evals" / "results" / "injection_holdout.json"
out.write_text(json.dumps(res, indent=2))

"""Injection detector: block rate on attacks, false-positive rate on benign. Usage: eval_injection.py [--test]"""
import argparse, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.injection import InjectionDetector  # noqa: E402
from evals.injection_set import DEV, TEST  # noqa: E402
from mcp_servers.rag_pipeline import parse_note  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
det = InjectionDetector()
data = TEST if a.test else DEV
vault = [(parse_note(Path(p).read_text())[1], "benign") for p in sorted(glob.glob("knowledge_vault/*.md"))]
for name, items in (("direct", [x for x in data if x[1] == "direct"]), ("indirect", [x for x in data if x[1] == "indirect"])):
    hit = [det.scan(t).is_attack for t, _ in items]
    print(f"{'TEST' if a.test else 'DEV'} {name:8s}: blocked {sum(hit)}/{len(hit)}")
    for (t, _), h in zip(items, hit):
        if not h: print("   MISSED:", repr(t[:90]))
ben = [x for x in data if x[1] == "benign"] + vault
fp = [(t, det.scan(t)) for t, _ in ben if det.scan(t).is_attack]
print(f"{'TEST' if a.test else 'DEV'} benign  : flagged {len(fp)}/{len(ben)} (incl. {len(vault)} knowledge-base chunks)")
for t, v in fp: print("   FALSE POSITIVE:", repr(t[:90]), v.family)

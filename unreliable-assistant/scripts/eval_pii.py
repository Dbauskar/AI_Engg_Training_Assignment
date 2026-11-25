"""PII recall / false-redaction / latency. Usage: uv run python scripts/eval_pii.py [--test] [--no-ner]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pii import PiiDetector  # noqa: E402
from evals.pii_set import CLEAN, DEV, TEST  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--no-ner", action="store_true"); a = ap.parse_args()
det = PiiDetector(use_ner=not a.no_ner)
data = TEST if a.test else DEV
det.redact("warm up John Smith")
leaks, total, lat = [], 0, []
for text, secrets in data:
    t = time.perf_counter(); r = det.redact(text); lat.append((time.perf_counter() - t) * 1000)
    for s in secrets:
        total += 1
        if s in r.text:
            leaks.append((s, r.text))
fr = [(c, det.redact(c).text) for c in CLEAN if det.redact(c).text != c]
print(f"{'TEST' if a.test else 'DEV'} {'(regex only)' if a.no_ner else '(regex + Presidio NER)'}: PII items redacted {total - len(leaks)}/{total} "
      f"= {(total - len(leaks)) / total:.1%} (KPI-02 >= 99.8%) | clean texts altered {len(fr)}/{len(CLEAN)} | latency p50 {np.percentile(lat, 50):.1f} p95 {np.percentile(lat, 95):.1f} ms")
for s, t in leaks: print("   LEAK:", repr(s), "->", repr(t))
for c, t in fr: print("   ALTERED:", repr(c), "->", repr(t))

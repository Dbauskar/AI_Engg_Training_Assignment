"""36-row KPI pass/fail table. Exits non-zero if any KPI FAILS or is NOT MEASURED.
  uv run python scripts/kpi_gate.py [--strict-scale] [--json out.json]
--strict-scale additionally fails on results that were only measured at reduced scale (use for a release at full scale).
Statuses: PASS | PASS (reduced scale) | UNVERIFIED (reduced scale) | FAIL | NOT MEASURED. Nothing is ever defaulted to PASS."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import kpi  # noqa: E402


def fmt(v, unit):
    if v is None:
        return "-"
    return f"{v:.4g}{(' ' + unit) if unit else ''}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-scale", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = kpi.evaluate()
    print(f"{'KPI':7} {'metric':42} {'target':>14} {'measured':>14}  {'status':27} scale")
    for r in rows:
        print(f"{r.kpi:7} {r.metric[:42]:42} {r.op + ' ' + fmt(r.threshold, r.unit):>14} {fmt(r.value, r.unit):>14}  {r.status:27} {r.scale}")
    s = kpi.summary(rows)
    print("\nSummary:", ", ".join(f"{k}={v}" for k, v in sorted(s.items())))
    reduced = [r.kpi for r in rows if "reduced scale" in r.status]
    if reduced:
        print(f"Measured at reduced scale only ({len(reduced)}): {', '.join(reduced)}")
    missing = [r.kpi for r in rows if r.status == "NOT MEASURED"]
    if missing:
        print(f"Not measured ({len(missing)}): {', '.join(missing)}")
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": s, "rows": kpi.as_dicts(rows)}, indent=2))
    bad = s.get("FAIL", 0) + s.get("NOT MEASURED", 0) + (len(reduced) if a.strict_scale else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

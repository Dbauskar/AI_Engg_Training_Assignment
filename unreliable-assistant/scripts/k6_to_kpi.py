"""Convert a k6 --summary-export/handleSummary JSON into KPI-31..36 (and per-stage server timings).
Usage: uv run python scripts/k6_to_kpi.py load_tests/out/k6_summary.json [--vus 500] [--label "local in-memory stack"]
Writes evals/results/load_k6.json and merges a "load" section into evals/results.json (run_ragas.py rewrites results.json, so the
scorecard and kpi_gate read the standalone file). KPI-36 (1-hour soak RSS growth) is NOT produced here: k6 cannot measure it."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def convert(summary: dict, vus: int, label: str) -> dict:
    m = summary["metrics"]
    dur = m["http_req_duration"]
    d = dur.get("values", dur)  # k6 handleSummary nests under "values" in older versions, flat in newer
    scale = f"{vus} VUs ({label})"
    reqs = m["http_reqs"].get("values", m["http_reqs"])
    r5 = m.get("http_5xx", {}).get("values", m.get("http_5xx", {}))
    kp = {
        "KPI-31": d["med"], "KPI-32": d["p(95)"], "KPI-33": d["p(99)"],
        "KPI-34": r5.get("rate", 0.0), "KPI-35": reqs["rate"],
    }
    stage = {}
    for name in ("srv_total_ms", "srv_hook_pre_ms", "srv_triage_ms", "srv_retrieval_ms", "srv_generation_ms"):
        if name in m:
            v = m[name].get("values", m[name])
            stage[name] = {"p50": v.get("med"), "p95": v.get("p(95)"), "p99": v.get("p(99)")}
    return {"scale": scale, "vus": vus, "requests": reqs["count"],
            "kpis": {k: {"value": v, "scale": scale} for k, v in kp.items()}, "server_stage_ms_under_load": stage,
            "not_measured": ["KPI-36 (needs a 1-hour soak with RSS sampling)"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--vus", type=int, default=None)
    ap.add_argument("--label", default="local in-memory stack, single process, 4 vCPU")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    vus = a.vus or s.get("setup_data", {}).get("vus_target") or int(s["metrics"]["vus_max"].get("values", s["metrics"]["vus_max"])["max"])
    out = convert(s, vus, a.label)
    (ROOT / "evals" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "evals" / "results" / "load_k6.json").write_text(json.dumps(out, indent=2))
    rp = ROOT / "evals" / "results.json"
    if rp.exists():
        d = json.loads(rp.read_text())
        d["load"] = out
        rp.write_text(json.dumps(d, indent=2))
    for k, v in out["kpis"].items():
        print(f"{k}: {v['value']:.4g}  [{out['scale']}]")
    print("KPI-36: not measured")

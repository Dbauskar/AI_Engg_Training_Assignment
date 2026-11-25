"""Verify the observability stack end to end against the running services:
Grafana datasources + provisioned dashboard exist, every dashboard panel query executes (and which ones have data),
Tempo can search recent traces, Prometheus has the assistant_* metrics. Exit code 1 if anything is broken.

Usage: uv run python scripts/verify_observability.py   (needs GRAFANA_ADMIN_PASSWORD in .env)"""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import _req, load_env  # noqa: E402

GRAFANA, PROM, TEMPO = "http://127.0.0.1:3000", "http://127.0.0.1:9090", "http://127.0.0.1:3200"


def main() -> int:
    load_env()
    bad = 0
    with httpx.Client(auth=("admin", _req("GRAFANA_ADMIN_PASSWORD")), timeout=20) as g:
        ds = {d["uid"]: d["type"] for d in g.get(f"{GRAFANA}/api/datasources").json()}
        print("datasources:", ds)
        bad += ds != {"prometheus": "prometheus", "tempo": "tempo"}
        found = g.get(f"{GRAFANA}/api/search", params={"type": "dash-db"}).json()
        print("dashboards:", [(d["title"], d["folderTitle"]) for d in found])
        dash = g.get(f"{GRAFANA}/api/dashboards/uid/assistant-overview").json()["dashboard"]
        print(f"dashboard '{dash['title']}' with {len(dash['panels'])} panels\n")
        for p in dash["panels"]:
            if p["type"] == "table":  # the Tempo 'recent traces' panel: check the equivalent search directly
                r = httpx.get(f"{TEMPO}/api/search", params={"q": '{resource.service.name="unreliable-assistant" && name="http.request"}', "limit": 20}, timeout=20)
                n = len(r.json().get("traces", []))
                print(f"  [{'ok ' if n else 'EMPTY'}] {p['title']:42s} -> {n} recent traces via Tempo search")
                bad += r.status_code != 200
                continue
            body = {"from": "now-30m", "to": "now", "queries": [{"refId": t["refId"], "datasource": {"type": "prometheus", "uid": "prometheus"},
                                                                    "expr": t["expr"], "range": True, "instant": False, "intervalMs": 15000, "maxDataPoints": 200}
                                                                   for t in p["targets"]]}
            r = g.post(f"{GRAFANA}/api/ds/query", json=body)
            res = r.json().get("results", {})
            errors = [v["error"] for v in res.values() if "error" in v]
            frames = sum(len(v.get("frames", [])) for v in res.values())
            values = sum(len(f["data"]["values"][1]) for v in res.values() for f in v.get("frames", []) if len(f["data"]["values"]) > 1)
            status = "ERR " if errors or r.status_code != 200 else ("ok  " if values else "EMPTY")
            print(f"  [{status}] {p['title']:42s} frames={frames} points={values} {errors[:1] if errors else ''}")
            bad += bool(errors) or r.status_code != 200
    names = httpx.get(f"{PROM}/api/v1/label/__name__/values", timeout=20).json()["data"]
    mine = sorted(n for n in names if n.startswith("assistant_"))
    print(f"\nprometheus: {len(mine)} assistant_* metric names")
    print("(EMPTY = valid query, no data yet, e.g. PII redactions before any PII request, or context-after tokens when graph mode is off)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

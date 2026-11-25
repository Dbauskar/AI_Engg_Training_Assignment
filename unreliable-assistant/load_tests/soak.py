"""Shortened soak for KPI-36: constant load on the local-stack server while sampling its RSS.
  uv run python -m load_tests.soak [--minutes 10] [--vus 100]
KPI-36 asks for RSS growth over a 1-HOUR soak. A shorter run is recorded with its true duration, so the scorecard shows it as
UNVERIFIED (or FAIL), never PASS. Growth is measured from the 2-minute mark (after warm-up and cache fill) to the end.
NOTE: the local stack's fake Redis never expires keys, so it can show growth a real Redis (TTL + maxmemory) would not."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rss_kb(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=10); ap.add_argument("--vus", type=int, default=100)
    ap.add_argument("--port", type=int, default=8081); a = ap.parse_args()
    srv = subprocess.Popen([sys.executable, "-m", "load_tests.local_server", "--port", str(a.port)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import httpx
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{a.port}/readyz", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        secs = int(a.minutes * 60)
        env = {**os.environ, "VUS": str(a.vus), "RAMP": "20", "HOLD": str(secs), "THINK": "1",
               "BASE_URL": f"http://127.0.0.1:{a.port}", "SUMMARY_OUT": str(ROOT / "load_tests/out/soak_summary.json")}
        k6 = subprocess.Popen(["k6", "run", "--quiet", "load_tests/k6_load.js"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0, samples = time.time(), []
        while k6.poll() is None:
            samples.append((time.time() - t0, rss_kb(srv.pid)))
            time.sleep(10)
        base = [r for t, r in samples if t >= 120]
        out = {"minutes": a.minutes, "vus": a.vus, "samples": len(samples), "rss_kb_start_of_window": base[0] if base else None,
               "rss_kb_end": samples[-1][1], "growth": (samples[-1][1] - base[0]) / base[0] if base else None, "series": samples[::6]}
        (ROOT / "load_tests/out/soak.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "series"}))
    finally:
        srv.terminate()

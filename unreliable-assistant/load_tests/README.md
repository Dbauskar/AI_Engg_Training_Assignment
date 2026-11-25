# Load tests (KPI-31..36)

## Local run (what has actually been measured)
```bash
uv run python -m load_tests.make_tokens --n 500 --csv     # tokens.json (k6) + tokens.csv (JMeter), git-ignored
uv run python -m load_tests.local_server --port 8080      # real pipeline on an in-memory Postgres/Redis stand-in (no docker needed)
VUS=500 HOLD=90 RAMP=45 k6 run load_tests/k6_load.js      # writes load_tests/out/k6_summary.json
uv run python scripts/k6_to_kpi.py load_tests/out/k6_summary.json   # KPI-31..35 -> evals/results/load_k6.json
```
With the real stack use `docker compose up -d` and `uvicorn app.main:app` instead of `local_server`.
Env: `VUS` (peak), `HOLD`/`RAMP` seconds, `THINK` seconds between requests, `BASE_URL`, `SUMMARY_OUT`.
Thresholds in the script are the design values: p50<250, p95<450, p99<600 ms, 5xx<0.05%.

## Distributed run (JMeter, 1 controller + N workers; 50,000 users)
1. Size it: ~50,000 JMeter threads need roughly 1,500-2,500 threads per worker, so 25-35 workers (4 vCPU / 8 GB each).
   Use the same JVM flags on every worker: `-Xms4g -Xmx4g -XX:+UseG1GC`.
2. On each worker: `jmeter-server -Djava.rmi.server.hostname=<worker-ip>` (open 1099 and the `server.rmi.localport`); copy
   `load_tests/tokens.csv` to the same path on every worker (the controller does not ship data files).
3. On the controller: 
   `jmeter -n -t load_tests/jmeter_plan.jmx -R w1,w2,... -Jthreads=1700 -Jramp=300 -Jhold=600 -Jhost=<lb> -Jport=443 -Jprotocol=https -l out/jmeter.jtl -e -o out/report`
   Note `threads` is PER WORKER; total VUs = threads x workers.
4. k6 alternative: k6-operator (`TestRun` with `parallelism: N`) or `k6 run --execution-segment` on N hosts; set `VUS` to the per-host share
   and sum the summaries. Raise `ulimit -n` and use `--no-connection-reuse=false` (keep-alive on).
5. Convert: `scripts/k6_to_kpi.py out/k6_summary.json --vus 50000 --label "cluster, N workers"`. Only a result with `50000 VUs` in
   its scale label can show PASS for KPI-31..36; anything smaller is shown as UNVERIFIED (reduced scale) or FAIL.
6. KPI-36 needs a separate 1-hour soak at steady load with `process_resident_memory_bytes` scraped by Prometheus; record the delta in
   `evals/results/load_k6.json` under `kpis.KPI-36`.

## Caveats
- The local server uses an in-memory state store and fake Redis, a single uvicorn process and 4 vCPU, with the embedding/rerank models on CPU.
  It measures application CPU/queueing only. It says nothing about Postgres, Redis, network or GPU serving.
- The 50,000-user / 8,500 RPS target needs a cluster; see deploy/ for the scale-out layout.

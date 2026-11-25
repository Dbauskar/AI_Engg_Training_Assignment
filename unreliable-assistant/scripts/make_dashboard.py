"""Generate the provisioned Grafana dashboard (deploy/grafana/dashboards/assistant.json). Metric names are the ones
Prometheus really exposes (checked against the running collector); thresholds come from docs/design.md section 5."""
import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
TEMPO = {"type": "tempo", "uid": "tempo"}
_id = 0


def panel(title, kind, targets, x, y, w=8, h=7, unit=None, thresholds=None, desc="", **extra):
    global _id
    _id += 1
    fc = {"defaults": {"unit": unit} if unit else {}, "overrides": []}
    if thresholds:
        fc["defaults"]["thresholds"] = {"mode": "absolute", "steps": [{"color": c, "value": v} for c, v in thresholds]}
        fc["defaults"]["color"] = {"mode": "thresholds"}
    return {"id": _id, "title": title, "type": kind, "datasource": extra.pop("ds", PROM), "description": desc,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "fieldConfig": fc, "targets": targets, **extra}


def q(expr, legend="", ref="A"):
    return {"refId": ref, "expr": expr, "legendFormat": legend, "datasource": PROM}


LAT = "assistant_request_latency_ms_bucket"
panels = [
    panel("Requests / s by outcome", "timeseries", [q("sum by (outcome) (rate(assistant_requests_total[1m]))", "{{outcome}}")], 0, 0, unit="reqps"),
    panel("End-to-end latency p50 / p95 / p99", "timeseries",
          [q(f"histogram_quantile(0.50, sum by (le) (rate({LAT}[5m])))", "p50", "A"),
           q(f"histogram_quantile(0.95, sum by (le) (rate({LAT}[5m])))", "p95", "B"),
           q(f"histogram_quantile(0.99, sum by (le) (rate({LAT}[5m])))", "p99", "C")], 8, 0, unit="ms",
          desc="Design SLA: p95 <= 450 ms, p99 <= 600 ms (KPI-32/33). Measured from the request start, so cache hits and blocked requests are included."),
    panel("p95 latency vs 450 ms SLA", "stat", [q(f"histogram_quantile(0.95, sum by (le) (rate({LAT}[5m])))", "p95")], 16, 0, w=4, unit="ms",
          thresholds=[("green", None), ("red", 450)]),
    panel("p99 latency vs 600 ms SLA", "stat", [q(f"histogram_quantile(0.99, sum by (le) (rate({LAT}[5m])))", "p99")], 20, 0, w=4, unit="ms",
          thresholds=[("green", None), ("red", 600)]),
    panel("Latency by route (p95)", "timeseries", [q(f"histogram_quantile(0.95, sum by (le, route) (rate({LAT}[5m])))", "{{route}}")], 0, 7, unit="ms"),
    panel("Non-answer outcomes / s", "timeseries",
          [q('sum by (outcome) (rate(assistant_requests_total{outcome=~"BLOCKED|REFUSED|ABSTAINED|DEGRADED|HALTED|FROZEN"}[1m]))', "{{outcome}}")], 8, 7, unit="reqps",
          desc="BLOCKED = injection/HITL/session-breaker; REFUSED = out of scope; ABSTAINED = not found or vetoed; FROZEN = awaiting human approval."),
    panel("Blocked by category / s", "timeseries", [q("sum by (category) (rate(assistant_blocked_total[5m]))", "{{category}}")], 16, 7, unit="reqps"),
    panel("Semantic cache hit ratio", "stat",
          [q("sum(rate(assistant_cache_hit_total[5m])) / (sum(rate(assistant_cache_hit_total[5m])) + sum(rate(assistant_cache_miss_total[5m])))", "hit ratio")],
          0, 14, w=4, unit="percentunit", thresholds=[("red", None), ("green", 0.35)], desc="KPI-27 target >= 35% in steady state."),
    panel("Cross-tenant cache leaks blocked", "stat", [q("sum(assistant_cache_leak_blocked_total) or vector(0)", "blocked")], 4, 14, w=4,
          thresholds=[("green", None), ("red", 1)], desc="Must stay 0 (KPI-28). Non-zero means an entry for another tenant/user was found and dropped."),
    panel("Bulkhead occupancy", "timeseries", [q("assistant_pool_in_flight / assistant_pool_size", "{{pool}}")], 8, 14, unit="percentunit",
          desc="In-flight / capacity per pool (rag=64, db=128, llm=256, triage=16). Sustained 100% means load is being shed."),
    panel("Requests shed by full pools / s", "timeseries", [q("sum by (pool) (rate(assistant_pool_rejected_total[1m]))", "{{pool}}")], 16, 14, unit="reqps"),
    panel("PII redactions by type / s", "timeseries", [q("sum by (type) (rate(assistant_pii_redacted_total[5m]))", "{{type}}")], 0, 21, unit="ops"),
    panel("Online quality proxies (avg)", "timeseries",
          [q("rate(assistant_quality_context_utilization_sum[5m]) / rate(assistant_quality_context_utilization_count[5m])", "context utilization", "A"),
           q("rate(assistant_quality_answer_relevancy_proxy_sum[5m]) / rate(assistant_quality_answer_relevancy_proxy_count[5m])", "answer relevancy (proxy)", "B")],
          8, 21, desc="Cheap stand-ins, NOT RAGAS: the real faithfulness/relevancy/precision are measured offline on the golden set."),
    panel("Context tokens before / after (avg)", "timeseries",
          [q("rate(assistant_context_tokens_before_sum[5m]) / rate(assistant_context_tokens_before_count[5m])", "before (raw chunks)", "A"),
           q("rate(assistant_context_tokens_after_sum[5m]) / rate(assistant_context_tokens_after_count[5m])", "after (graph context)", "B")],
          16, 21, desc="'after' only has data when GRAPH_MODE is measure or use."),
    panel("Recent traces", "table",
          [{"refId": "A", "datasource": TEMPO, "queryType": "traceql", "query": '{resource.service.name="unreliable-assistant" && name="http.request"}', "limit": 20, "tableType": "traces"}],
          0, 28, w=24, h=9, ds=TEMPO, desc="Open a trace id to see the full span tree: hook.pre, cache.lookup, triage.classify, mcp.rag.*, agent.react.step, llm.generate, hook.post.ragas."),
]
dash = {"uid": "assistant-overview", "title": "Unreliable Assistant - Overview", "tags": ["assistant", "slo"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "refresh": "10s", "time": {"from": "now-30m", "to": "now"}, "editable": False, "panels": panels,
        "templating": {"list": []}, "annotations": {"list": []}}
out = Path(__file__).resolve().parents[1] / "deploy/grafana/dashboards/assistant.json"
out.write_text(json.dumps(dash, indent=2))
print("wrote", out.relative_to(out.parents[3]), f"({len(panels)} panels)")

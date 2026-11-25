"""OpenTelemetry wiring: traces and metrics exported over OTLP to the collector (-> Tempo / Prometheus -> Grafana).

`setup_telemetry` installs the GLOBAL providers, which can only be set once per process, so it is called from the real
entry point only (never from tests, which install in-memory providers). `register_runtime_metrics` exposes state that is
not naturally an event: bulkhead occupancy/rejections and cache counters.
"""
from __future__ import annotations

from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.resources import Resource


def setup_telemetry(service_name: str, otlp_endpoint: str, version: str = "0.1.0", export_interval_ms: int = 5000) -> tuple[Any, Any]:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name, "service.version": version})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True, timeout=3)))
    mp = MeterProvider(resource=resource, metric_readers=[
        PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True, timeout=3),
                                      export_interval_millis=export_interval_ms)])
    trace.set_tracer_provider(tp)
    metrics.set_meter_provider(mp)
    return tp, mp


def register_runtime_metrics(meter: Any, pools: Any, cache: Any | None = None) -> None:
    def pool_obs(field: str):
        def cb(_: CallbackOptions):
            return [Observation(p.stats()[field], {"pool": p.name}) for p in (pools.rag, pools.db, pools.llm, pools.triage)]
        return cb

    meter.create_observable_gauge("assistant.pool.in_flight", callbacks=[pool_obs("in_flight")], description="requests currently holding a slot")
    meter.create_observable_gauge("assistant.pool.size", callbacks=[pool_obs("size")])
    meter.create_observable_counter("assistant.pool.rejected", callbacks=[pool_obs("rejected")], description="requests shed because the pool was full")
    if cache is not None:
        def cache_obs(kind: str):
            def cb(_: CallbackOptions):
                st = cache.stats
                return [Observation({"hit": st.hits_exact + st.hits_semantic, "miss": st.misses, "bypass": st.bypassed,
                                     "error": st.errors, "leak_blocked": st.leaks_blocked, "rejected_pii": st.rejected_pii}[kind])]
            return cb

        for kind in ("hit", "miss", "bypass", "error", "leak_blocked", "rejected_pii"):
            meter.create_observable_counter(f"assistant.cache.{kind}", callbacks=[cache_obs(kind)])

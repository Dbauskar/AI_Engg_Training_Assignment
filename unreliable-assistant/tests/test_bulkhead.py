import asyncio
import threading
import time
import uuid

import pytest
from mcp.client import Client

from app.bulkhead import Bulkhead, BulkheadFull, Pools
from app.health import health_snapshot
from mcp_servers.rag_server import build_server
from tests.test_rag_pipeline import TENANT, needs_models, retriever  # noqa: F401  (session fixture)


def test_design_pool_sizes():
    p = Pools.from_env()
    assert (p.rag.size, p.db.size, p.llm.size) == (64, 128, 256)  # design Gate 9
    assert p.triage.size > 0


async def test_concurrency_is_capped_and_overflow_fails_fast():
    b = Bulkhead("rag", 3, queue_timeout_s=0.05)
    running, peak = 0, 0
    lock = threading.Lock()

    def work():
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.2)
        with lock:
            running -= 1
        return "ok"

    tasks = [asyncio.create_task(b.run_sync(work)) for _ in range(6)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(r == "ok" for r in results) == 3 and sum(isinstance(r, BulkheadFull) for r in results) == 3
    assert peak == 3 and b.rejected == 3 and b.peak == 3
    assert time.perf_counter() - t0 < 0.5  # rejected callers did not wait for the slow ones
    assert b.in_flight == 0


async def test_blocking_work_runs_on_the_pools_own_threads_and_slot_is_released_on_error():
    b = Bulkhead("db", 2)
    name = await b.run_sync(lambda: threading.current_thread().name)
    assert name.startswith("bh-db")
    with pytest.raises(ZeroDivisionError):
        await b.run_sync(lambda: 1 / 0)
    assert b.in_flight == 0 and b.completed == 2
    assert await b.run_sync(lambda: 7) == 7  # still usable


async def test_health_stays_fast_and_loop_stays_responsive_while_rag_pool_is_saturated():
    pools = Pools(Bulkhead("rag", 4), Bulkhead("db", 8), Bulkhead("llm", 8), Bulkhead("triage", 4))
    lag = []

    async def ticker():  # event-loop responsiveness probe
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.005)
            lag.append(time.perf_counter() - t - 0.005)

    tick = asyncio.create_task(ticker())
    slow = [asyncio.create_task(pools.rag.run_sync(time.sleep, 0.6)) for _ in range(4)]  # pool now full
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    snap = await health_snapshot(pools)
    health_ms = (time.perf_counter() - t0) * 1000
    assert health_ms < 20 and snap["status"] == "degraded" and snap["saturated_pools"] == ["rag"]
    with pytest.raises(BulkheadFull):  # the 5th retrieval is shed immediately...
        await pools.rag.run_sync(lambda: None)
    assert await pools.triage.run_sync(lambda: "routed") == "routed"  # ...while triage and db are unaffected
    assert await pools.db.run_sync(lambda: "row") == "row"
    await asyncio.gather(*slow)
    tick.cancel()
    assert max(lag) < 0.05, f"event loop stalled {max(lag) * 1000:.0f} ms"
    assert (await health_snapshot(pools))["status"] == "ok"


@needs_models
async def test_slow_reranker_does_not_block_health_checks_through_the_real_server(retriever):  # noqa: F811
    """Real HybridRetriever behind the real MCP server, reranker made slow. Pool of 4, 8 concurrent calls."""
    pools = Pools(Bulkhead("rag", 4, 0.05), Bulkhead("db", 8), Bulkhead("llm", 8), Bulkhead("triage", 4))
    original = retriever.reranker.score

    def slow_score(query, passages):
        time.sleep(0.6)
        return original(query, passages)

    retriever.reranker.score = slow_score
    try:
        args = lambda: {"tenant_id": TENANT, "user_id": "u1", "request_id": str(uuid.uuid4()),  # noqa: E731
                        "query": "How many vacation days do I get each year?"}
        async with Client(build_server(retriever, pools=pools)) as client:
            calls = [asyncio.create_task(client.call_tool("hybrid_retrieve", args())) for _ in range(8)]
            await asyncio.sleep(0.25)  # the four admitted calls are now stuck inside the slow reranker
            t0 = time.perf_counter()
            snap = await health_snapshot(pools)
            health_ms = (time.perf_counter() - t0) * 1000
            results = await asyncio.gather(*calls)
        assert health_ms < 20 and "rag" in snap["saturated_pools"]
        ok = [r for r in results if not r.is_error]
        shed = [r for r in results if r.is_error and "E_RATE_LIMITED" in r.content[0].text]
        assert len(ok) == 4 and len(shed) == 4  # excess load was shed with a retryable error, not queued
        assert pools.rag.peak == 4
    finally:
        retriever.reranker.score = original


async def test_pooled_work_stays_inside_the_requests_trace(spans):
    """Regression: run_in_executor does not copy contextvars, which silently started a new trace inside every pool call."""
    from opentelemetry import trace

    tracer = trace.get_tracer("test")
    b = Bulkhead("rag", 2)

    def work():
        with tracer.start_as_current_span("inside-pool"):
            pass

    with tracer.start_as_current_span("request") as root:
        await b.run_sync(work)
    inner = next(s for s in spans.get_finished_spans() if s.name == "inside-pool")
    assert inner.context.trace_id == root.get_span_context().trace_id
    assert inner.parent.span_id == root.get_span_context().span_id

"""Chaos scenarios from the FMEA, run as regression tests."""
import pytest

from evals import chaos as K


async def test_infinite_loops_always_halt(spans):
    r = await K.sku_loop()
    assert r["status"] == "PASS", r["failed"]
    assert any(s.name == "agent.react.run" for s in spans.get_finished_spans())


async def test_no_cross_tenant_cache_leaks():
    r = await K.cross_tenant_cache()
    assert r["status"] == "PASS" and r["leaks"] == 0 and r["legitimate_hits"] > 0


async def test_hung_retrieval_tool_is_bounded_and_cancelled():
    r = await K.mcp_hang()
    assert r["status"] == "PASS", r


async def test_run_all_emits_a_span_per_scenario(spans):
    await K.run_all()
    names = [s.attributes["chaos.scenario"] for s in spans.get_finished_spans() if s.name == "chaos.scenario"]
    assert names == ["sku_loop", "cross_tenant_cache", "redis_down", "mcp_hang", "llm_429"]


async def test_redis_down_cache_bypass_still_holds():
    r = await K.redis_down()
    assert r["cache_get_when_down"] == "bypass" and r["answer_outcome_with_cache_down"] == "ANSWERED" and r["tier3_submit_handler_runs"] <= 1


async def test_redis_down_tier3_degrades_gracefully():
    assert (await K.redis_down())["status"] == "PASS"


async def test_llm_429_is_retried_then_degrades():
    assert (await K.llm_429())["status"] == "PASS"

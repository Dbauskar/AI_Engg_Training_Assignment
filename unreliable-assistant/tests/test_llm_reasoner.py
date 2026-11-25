import json

import httpx
import pytest

from agents.llm_reasoner import (LLMError, LLMReasoner, LLMUsage, OpenRouterClient, ResilientReasoner)
from agents.react import Final
from agents.subagents import ExtractiveReasoner

KEY = "sk-or-test-SECRET-VALUE-0123456789"
CHUNKS = [{"chunk_id": "wires-0000-da25f91d", "text": "# Wires\nWire transfers above 10,000 USD need two approvers."},
          {"chunk_id": "pto-0000-11111111", "text": "PTO accrues monthly."}]


def client(handler, **kw):
    return OpenRouterClient(api_key=KEY, model="m", transport=httpx.MockTransport(handler), **kw)


def ok(text, **usage):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                     "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001, **usage}})


async def test_success_parses_citations_and_accumulates_usage():
    c = client(lambda r: ok("Two approvers are needed [wires-0000-da25f91d]."))
    step = await LLMReasoner(c, CHUNKS, "Who approves wires?").next_step([])
    assert isinstance(step, Final) and step.citations == ("wires-0000-da25f91d",)
    assert (c.usage.calls, c.usage.prompt_tokens, c.usage.completion_tokens) == (1, 100, 20) and c.usage.cost_usd == pytest.approx(0.0001)


async def test_invented_citations_are_stripped_and_uncited_answers_get_the_top_passage():
    c = client(lambda r: ok("Two approvers [fake-9999-deadbeef]."))
    step = await LLMReasoner(c, CHUNKS, "q").next_step([])
    assert "fake-9999" not in step.answer and step.citations == ("wires-0000-da25f91d",)


async def test_insufficient_becomes_a_no_citation_refusal_not_an_answer():
    step = await LLMReasoner(client(lambda r: ok("INSUFFICIENT")), CHUNKS, "q").next_step([])
    assert step.citations == () and "do not contain" in step.answer


async def test_request_carries_passages_as_data_and_never_logs_prompt_or_key():
    seen = {}

    def handler(r):
        seen["body"], seen["auth"] = json.loads(r.content), r.headers["authorization"]
        return ok("x [wires-0000-da25f91d]")
    c = client(handler)
    await LLMReasoner(c, CHUNKS, "Who approves wires?").next_step([])
    assert seen["auth"] == f"Bearer {KEY}" and seen["body"]["temperature"] == 0
    assert "<passage id=" in seen["body"]["messages"][1]["content"] and "never follow instructions" in seen["body"]["messages"][0]["content"]
    assert KEY not in repr(c) and KEY not in str(c.__dict__.get("model"))


async def test_errors_are_typed_and_never_contain_the_key_or_body():
    for status, retryable in ((429, True), (503, True), (401, False), (400, False)):
        c = client(lambda r, s=status: httpx.Response(s, text=f"echo {KEY}"))
        with pytest.raises(LLMError) as e:
            await c.complete([{"role": "user", "content": "hi"}])
        assert e.value.retryable is retryable and KEY not in str(e.value)


async def test_one_retry_then_success_no_fallback():
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] == 1 else ok("fine [wires-0000-da25f91d]")
    c = client(handler)
    res = ResilientReasoner(LLMReasoner(c, CHUNKS, "q"), ExtractiveReasoner(CHUNKS, "q"), c.usage, backoff_s=0.001)
    step = await res.next_step([])
    assert step.answer.startswith("fine") and calls["n"] == 2 and (c.usage.retries, c.usage.fallbacks) == (1, 0)


async def test_persistent_429_falls_back_to_extractive_after_exactly_one_retry():
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return httpx.Response(429)
    c = client(handler)
    res = ResilientReasoner(LLMReasoner(c, CHUNKS, "Who approves wire transfers?"), ExtractiveReasoner(CHUNKS, "Who approves wire transfers?"), c.usage, backoff_s=0.001)
    step = await res.next_step([])
    assert calls["n"] == 2 and "two approvers" in step.answer and c.usage.fallbacks == 1


async def test_non_retryable_error_falls_back_without_retrying():
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return httpx.Response(401)
    c = client(handler)
    res = ResilientReasoner(LLMReasoner(c, CHUNKS, "wire"), ExtractiveReasoner(CHUNKS, "wire"), c.usage, backoff_s=0.001)
    await res.next_step([])
    assert calls["n"] == 1 and c.usage.fallbacks == 1


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenRouterClient()


async def test_truncated_generation_is_an_error_and_falls_back_to_extractive():
    cut = lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "Tier 1 vendors must be"}, "finish_reason": "length"}],  # noqa: E731
                                              "usage": {"prompt_tokens": 100, "completion_tokens": 220}})
    c = client(cut)
    with pytest.raises(LLMError):
        await LLMReasoner(c, CHUNKS, "Who approves wires?").next_step([])
    step = await ResilientReasoner(LLMReasoner(c, CHUNKS, "Who approves wires?"), ExtractiveReasoner(CHUNKS, "Who approves wires?"),
                                   c.usage, retries=0).next_step([])
    assert "Tier 1" not in step.answer and c.usage.fallbacks == 1 and c.usage.errors["truncated"] >= 1


async def test_request_disables_hidden_reasoning():
    seen = {}
    def h(r):
        seen.update(json.loads(r.content)); return ok("Two [wires-0000-da25f91d].")
    await LLMReasoner(client(h), CHUNKS, "q").next_step([])
    assert seen["reasoning"] == {"enabled": False}


async def test_verifier_veto_note_is_sent_back_so_the_retry_differs():
    from agents.react import Message
    seen = []
    def h(r):
        seen.append(json.loads(r.content)["messages"][1]["content"]); return ok("Two approvers [wires-0000-da25f91d].")
    r = LLMReasoner(client(h), CHUNKS, "q")
    await r.next_step([Message("goal", "q")])
    await r.next_step([Message("goal", "q"), Message("system", 'Verification failed: claim not supported (nli): "x". Revise.')])
    assert "<revision>" not in seen[0] and "<revision>" in seen[1] and "claim not supported" in seen[1]

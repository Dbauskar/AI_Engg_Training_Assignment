"""Chaos suite: five failure injections against the in-process stack. LOCAL ONLY (nothing here opens a network socket
except an optional ping to localhost Redis).

  uv run python -m evals.chaos [--out evals/results/chaos.json]

Scenarios (each reports what was OBSERVED against the fallback promised in docs/design.md section 6, FMEA):
  sku_loop              KPI-16  endless tool loops (same bad call, alternating calls, ever-new SKUs, denied tool, hung reasoner): must halt
  cross_tenant_cache    KPI-28  5 tenants x 4 users hammer the same query texts concurrently + planted foreign/unverified entries: zero leaks
  redis_down            FMEA 1  cache bypass, answers still served, and a Tier-3 submit under a dead idempotency store is never run twice
  mcp_hang              FMEA 3  a retrieval tool that never returns: bounded latency, graceful answer, hung call cancelled
  llm_429               FMEA 4  reasoner raises a rate-limit error: promised fallback is retry once, then extractive quotes

Redis: a real local Redis is used when `redis.ping()` works, otherwise evals/fake_redis.py (which only exercises the application's
own scoping and fallback logic). The report says which was used. Exit code is non-zero if any scenario FAILs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from opentelemetry import trace  # noqa: E402

from agents.llm_reasoner import LLMUsage, ResilientReasoner  # noqa: E402
from agents.pipeline import Pipeline  # noqa: E402
from agents.react import Act, Final, ReActAgent, ReActConfig, State  # noqa: E402
from app import contracts as C  # noqa: E402
from app.cache import SemanticCache  # noqa: E402
from app.harness import Caller, RedisIdempotencyStore, Tier, ToolExecutionHarness, ToolFailure, ToolSpec  # noqa: E402
from app.pii import PiiDetector  # noqa: E402
from evals.fake_redis import FakeRedis  # noqa: E402
from evals.redteam import TENANT, Target, _Triage  # noqa: E402

_tracer = trace.get_tracer("unreliable_assistant.chaos")
QUERY = "Who approves wires?"
SUBMIT = {"action": "TRANSFER", "account_id": "ACC-12345678", "amount": "100.50", "currency": "USD", "justification": "Quarterly settlement"}
CALLER = Caller(TENANT, "chaos", Tier.TIER_3_MUTATE)
SESSION = str(uuid.uuid4())


class RateLimited(Exception):
    """Stands in for an LLM provider's HTTP 429."""

    status_code = 429


def retrieve_harness(behaviour: str, counts: dict[str, int]) -> ToolExecutionHarness:
    h = ToolExecutionHarness()

    async def retrieve(m, c):
        counts["retrieve"] = counts.get("retrieve", 0) + 1
        if behaviour == "fail":
            raise ToolFailure("E_UPSTREAM_UNAVAILABLE", "retrieval down")
        if behaviour == "hang":
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                counts["cancelled"] = counts.get("cancelled", 0) + 1
                raise
        return C.HybridRetrieveOut(abstain=True, chunks=[], top_score=0.0, stage_latency_ms={"bm25": 1.0, "dense": 1.0, "rrf": 0.1, "rerank": 1.0})

    async def submit(m, c):
        counts["submit"] = counts.get("submit", 0) + 1
        return C.SubmitTransactionOut(status="PENDING_APPROVAL", approval_id=str(uuid.uuid4()), audit_hash="ab" * 32)

    h.register(ToolSpec("hybrid_retrieve", Tier.TIER_1_READ, C.HybridRetrieveIn, retrieve))
    h.register(ToolSpec("submit_transaction", Tier.TIER_3_MUTATE, C.SubmitTransactionIn, submit))
    return h


class Script:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    async def next_step(self, prompt):
        self.n += 1
        return await self.fn(self.n)


def agent(h, reasoner, **cfg) -> ReActAgent:
    return ReActAgent(h, _Triage(), reasoner, config=ReActConfig(**cfg) if cfg else None)


# ---------------------------------------------------------------------------------------------------------------------
async def sku_loop() -> dict[str, Any]:
    async def hang(n):
        await asyncio.sleep(5)

    variants: list[tuple[str, str, Any]] = []  # (name, handler behaviour, reasoner fn)
    for sku in (1, 7, 42, 99, 205, 310):
        variants.append((f"same_invalid_args_sku{sku}", "ok", lambda n, s=sku: _ret(Act("hybrid_retrieve", {"query": f"SKU-{s:03d}", "rerank_k": 99}))))
        variants.append((f"same_failing_call_sku{sku}", "fail", lambda n, s=sku: _ret(Act("hybrid_retrieve", {"query": f"SKU-{s:03d} warranty"}))))
        variants.append((f"alternate_two_failing_sku{sku}", "fail", lambda n, s=sku: _ret(Act("hybrid_retrieve", {"query": f"SKU-{s + (n % 2):03d} refund"}))))
        variants.append((f"ever_new_sku_ok_sku{sku}", "ok", lambda n, s=sku: _ret(Act("hybrid_retrieve", {"query": f"SKU-{s + n:03d} warranty"}))))
        variants.append((f"ever_new_sku_failing_sku{sku}", "fail", lambda n, s=sku: _ret(Act("hybrid_retrieve", {"query": f"SKU-{s + n:03d} warranty"}))))
    variants.append(("denied_tool_repeated", "ok", lambda n: _ret(Act("submit_transaction", SUBMIT))))  # not allowed on the RAG route
    variants.append(("hung_reasoner", "ok", hang))
    variants.append(("unknown_tool_repeated", "ok", lambda n: _ret(Act("delete_everything", {}))))

    rows = []
    for name, behaviour, fn in variants:
        counts: dict[str, int] = {}
        t0 = time.perf_counter()
        try:
            res = await agent(retrieve_harness(behaviour, counts), Script(fn)).run(CALLER, QUERY, SESSION)
            ms = (time.perf_counter() - t0) * 1000
            escaped = res.state in (State.HALTED, State.DONE) and res.iterations <= 5 and res.tool_calls <= 5 and ms < 1000
            rows.append({"variant": name, "escaped": escaped, "halt": res.halt_reason, "iterations": res.iterations, "tool_calls": res.tool_calls, "ms": round(ms, 1)})
        except Exception as e:  # noqa: BLE001
            rows.append({"variant": name, "escaped": False, "halt": f"EXCEPTION {type(e).__name__}", "iterations": None, "tool_calls": None, "ms": None})
    esc = sum(r["escaped"] for r in rows) / len(rows)
    return {"kpi": "KPI-16 infinite-loop escape rate", "threshold": 1.0, "value": esc, "status": "PASS" if esc == 1.0 else "FAIL", "n": len(rows),
            "halt_reasons": {k: sum(r["halt"] == k for r in rows) for k in {r["halt"] for r in rows}}, "failed": [r for r in rows if not r["escaped"]]}


async def _ret(x):
    return x


# ---------------------------------------------------------------------------------------------------------------------
def fake_embed(text: str) -> np.ndarray:
    v = np.zeros(64, dtype=np.float32)
    for w in text.lower().split():
        v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % 64] += 1.0
    return v / (np.linalg.norm(v) or 1.0)


async def make_redis() -> tuple[Any, str]:
    try:
        import redis.asyncio as aioredis
        r = aioredis.Redis(socket_timeout=1)
        await r.ping()
        await r.flushdb()
        return r, "real redis (localhost:6379, db flushed)"
    except Exception:  # noqa: BLE001
        return FakeRedis(), "evals/fake_redis.py (no Redis reachable: proves application-level scoping only)"


async def cross_tenant_cache() -> dict[str, Any]:
    redis, backend = await make_redis()
    cache = SemanticCache(redis, fake_embed, pii=PiiDetector(use_ner=False))
    tenants = [f"tenant-{c}" for c in "abcde"]
    users = [f"user{i}" for i in range(4)]
    texts = [f"What is the rule number {i} for wire approvals above limit {i}?" for i in range(12)]
    await asyncio.gather(*[cache.put(t, u, q, f"MARK<{t}|{u}|{i}>", verified=True) for t in tenants for u in users for i, q in enumerate(texts)])

    async def probe(t, u, q):
        r = await cache.get(t, u, q)
        return t, u, r

    results = await asyncio.gather(*[probe(t, u, q) for t in tenants for u in users for q in texts for _ in range(2)])
    leaks = [(t, u) for t, u, r in results if r.response and f"<{t}|{u}|" not in r.response]
    hits = sum(bool(r.response) for _, _, r in results)

    # planted entries: a foreign-tenant entry and an unverified entry stored under the victim's own exact key
    from app.cache import entry_key
    victim_q = "Planted question about the wire approval limit"
    k = entry_key("tenant-a", "user0", victim_q)
    await redis.set(k, json.dumps({"tenant_id": "tenant-b", "user_id": "user0", "verified": True, "response": "FOREIGN"}))
    r1 = await cache.get("tenant-a", "user0", victim_q)
    k2 = entry_key("tenant-a", "user1", victim_q)
    await redis.set(k2, json.dumps({"tenant_id": "tenant-a", "user_id": "user1", "verified": False, "response": "POISON"}))
    r2 = await cache.get("tenant-a", "user1", victim_q)
    unverified_put = await cache.put("tenant-a", "user2", "another question", "wrong answer", verified=False)
    pii_put = await cache.put("tenant-a", "user2", "pii question", "Contact jane.doe@example.com for details", verified=True)
    try:
        await cache.get("", "user0", "x")
        missing_tenant_rejected = False
    except Exception:  # noqa: BLE001
        missing_tenant_rejected = True
    ok = (not leaks and hits > 0 and not r1.response and not r2.response and not unverified_put and not pii_put and missing_tenant_rejected)
    return {"kpi": "KPI-28 cross-tenant cache leaks", "backend": backend, "probes": len(results), "legitimate_hits": hits, "leaks": len(leaks),
            "foreign_entry_served": bool(r1.response), "unverified_entry_served": bool(r2.response), "unverified_put_accepted": unverified_put,
            "pii_put_accepted": pii_put, "missing_tenant_rejected": missing_tenant_rejected, "leaks_blocked_counter": cache.stats.leaks_blocked, "status": "PASS" if ok else "FAIL"}


# ---------------------------------------------------------------------------------------------------------------------
async def redis_down() -> dict[str, Any]:
    from redis.exceptions import ConnectionError as RedisConnectionError

    redis = FakeRedis()
    cache = SemanticCache(redis, fake_embed, pii=PiiDetector(use_ner=False))
    await cache.put(TENANT, "chaos", QUERY, "cached answer", verified=True)
    redis.down = True
    got = await cache.get(TENANT, "chaos", QUERY)
    put = await cache.put(TENANT, "chaos", "other question", "answer", verified=True)
    t = Target(PiiDetector(use_ner=False))
    pipe = Pipeline(t.pipeline.orch, t.registry, cache)
    res = await pipe.answer(t.caller, "Who has to approve a wire transfer above 10,000 USD?", str(uuid.uuid4()))

    # Tier-3 submit with a dead idempotency store: it must never execute twice; an unhandled exception is a failure of graceful degradation
    counts: dict[str, int] = {}
    h = retrieve_harness("ok", counts)
    h._store = RedisIdempotencyStore(redis)  # noqa: SLF001 - swap in the dead store
    args = {"tenant_id": TENANT, "user_id": "chaos", "request_id": str(uuid.uuid4()), "session_id": SESSION, **SUBMIT}
    outcomes = []
    for _ in range(2):
        try:
            r = await h.call("submit_transaction", args, CALLER)
            outcomes.append("ok" if r.ok else f"error:{r.error.code}")
        except Exception as e:  # noqa: BLE001
            outcomes.append(f"EXCEPTION {type(e).__name__}")
    unhandled = any(o.startswith("EXCEPTION") for o in outcomes)
    ok = got.kind == "bypass" and put is False and res.outcome == "ANSWERED" and counts.get("submit", 0) <= 1 and not unhandled
    return {"fmea": "#1 Redis down", "cache_get_when_down": got.kind, "cache_put_accepted": put, "answer_outcome_with_cache_down": res.outcome,
            "tier3_submit_handler_runs": counts.get("submit", 0), "tier3_submit_outcomes": outcomes, "status": "PASS" if ok else "FAIL",
            "note": "fail-closed (error) is acceptable for Tier 3; an unhandled exception or a double execution is not"}


async def mcp_hang() -> dict[str, Any]:
    counts: dict[str, int] = {}
    script = Script(lambda n: _ret(Act("hybrid_retrieve", {"query": QUERY})))
    t0 = time.perf_counter()
    try:
        res = await agent(retrieve_harness("hang", counts), script).run(CALLER, QUERY, SESSION)
        ms = (time.perf_counter() - t0) * 1000
        await asyncio.sleep(0.05)
        ok = ms < 700 and res.state == State.HALTED and bool(res.answer) and counts.get("cancelled", 0) == counts.get("retrieve", 0)
        return {"fmea": "#3 MCP hang", "ms": round(ms, 1), "state": res.state.value, "halt": res.halt_reason, "tool_calls": res.tool_calls,
                "handler_started": counts.get("retrieve", 0), "handler_cancelled": counts.get("cancelled", 0), "status": "PASS" if ok else "FAIL"}
    except Exception as e:  # noqa: BLE001
        return {"fmea": "#3 MCP hang", "status": "FAIL", "exception": type(e).__name__}


async def llm_429() -> dict[str, Any]:
    """FMEA #4 promises: retry once with jittered backoff, then fall back to extractive quotes. Observed behaviour is reported."""
    rows = {}
    for name, fails in (("single_429_then_ok", 1), ("persistent_429", 10 ** 6)):
        script = Script(lambda n, f=fails: _raise_or_final(n, f))
        fallback = Script(lambda n: _fallback_final())
        usage = LLMUsage()
        resilient = ResilientReasoner(script, fallback, usage, retries=1, backoff_s=0.01)  # the wrapper the runtime really uses
        try:
            res = await agent(retrieve_harness("ok", {}), resilient).run(CALLER, QUERY, SESSION)
            rows[name] = {"raised": False, "state": res.state.value, "halt": res.halt_reason, "reasoner_calls": script.n,
                          "retries": usage.retries, "fallbacks": usage.fallbacks}
        except Exception as e:  # noqa: BLE001
            rows[name] = {"raised": True, "exception": type(e).__name__, "reasoner_calls": script.n}
    # Wrapper-level contract (the chaos agent's fake retrieval always abstains, so the loop itself may legitimately halt):
    # exactly one retry; a transient 429 never reaches the fallback; a persistent one does, once per generation step.
    direct = {}
    for name, fails in (("single_429_then_ok", 1), ("persistent_429", 10 ** 6)):
        sc, usage = Script(lambda n, f=fails: _raise_or_final(n, f)), LLMUsage()
        step = await ResilientReasoner(sc, Script(lambda n: _fallback_final()), usage, retries=1, backoff_s=0.01).next_step([])
        direct[name] = {"calls": sc.n, "retries": usage.retries, "fallbacks": usage.fallbacks, "got_answer": isinstance(step, Final)}
    for k, v in direct.items():
        rows[k]["wrapper"] = v
    ok = (all(not r["raised"] for r in rows.values())                       # the agent loop never sees a provider exception
          and direct["single_429_then_ok"] == {"calls": 2, "retries": 1, "fallbacks": 0, "got_answer": True}
          and direct["persistent_429"] == {"calls": 2, "retries": 1, "fallbacks": 1, "got_answer": True})
    return {"fmea": "#4 LLM 429", "variants": rows, "status": "PASS" if ok else "FAIL",
            "note": "ResilientReasoner (agents/llm_reasoner.py): one jittered retry, then the extractive fallback; provider 429 simulated"}


async def _fallback_final():
    return Final("Wires above 10,000 USD need two approvers. [wires-0000-da25f91d]", ())


async def _raise_or_final(n: int, fails: int):
    if n <= fails:
        raise RateLimited("429 Too Many Requests")
    return Final("Wires above 10,000 USD need two approvers.", ())


# ---------------------------------------------------------------------------------------------------------------------
async def run_all() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fn in (("sku_loop", sku_loop), ("cross_tenant_cache", cross_tenant_cache), ("redis_down", redis_down), ("mcp_hang", mcp_hang), ("llm_429", llm_429)):
        with _tracer.start_as_current_span("chaos.scenario") as span:
            span.set_attribute("chaos.scenario", name)
            out[name] = await fn()
            span.set_attribute("chaos.status", out[name]["status"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "evals/results/chaos.json"))
    a = ap.parse_args()
    res = asyncio.run(run_all())
    report = {"target": "in-process stack, fake tool handlers", "scenarios": res, "all_pass": all(r["status"] == "PASS" for r in res.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    for n, r in res.items():
        print(f"[{r['status']}] {n}: " + json.dumps({k: v for k, v in r.items() if k not in ('status', 'failed', 'variants', 'note')}, default=str)[:300])
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

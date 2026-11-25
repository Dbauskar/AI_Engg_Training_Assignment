"""Request pipeline: pre-plugins -> cache -> triage/subagents (orchestrator) -> post-plugins -> background cache fill.

Order matches docs/design.md section 3.2. The cache is consulted only AFTER the pre-plugins, so it is keyed on the redacted,
injection-checked query; a cache hit skips triage/retrieval/generation but still passes through the post-plugins (egress PII
scan, HITL invariant, metrics)."""
from __future__ import annotations

import asyncio
import time

from agents.subagents import AnswerResult, Orchestrator
from app.cache import SemanticCache
from app.harness import Caller
from plugins import REQUIRED, PluginRegistry, RequestContext

BLOCKED_ROUTE = "BLOCKED"


class Pipeline:
    def __init__(self, orchestrator: Orchestrator, registry: PluginRegistry, cache: SemanticCache | None = None) -> None:
        registry.require(REQUIRED)  # fail closed at startup: no service without the mandatory controls
        self.orch, self.reg, self.cache = orchestrator, registry, cache
        self._bg: set[asyncio.Task] = set()

    async def answer(self, caller: Caller, query: str, session_id: str) -> AnswerResult:
        t0 = time.perf_counter()
        ctx = RequestContext.create(caller.tenant_id, caller.user_id, session_id, query)
        del query  # from here on only ctx.query (redacted by the pii_redactor) is used downstream
        await self.reg.run_pre(ctx)
        hit = None
        if ctx.blocked is not None:
            res = AnswerResult(ctx.blocked.public_message, BLOCKED_ROUTE, "BLOCKED", reason=ctx.blocked.code)
        else:
            if self.cache is not None:
                hit = await self.cache.get(caller.tenant_id, caller.user_id, ctx.query)
            if hit is not None and hit.hit:
                res = AnswerResult(hit.response or "", "HYBRID_RAG", "ANSWERED", cache_hit=True, trace=["CACHE"])
            else:
                res = await self.orch.answer(caller, ctx.query, session_id, lambda data: self.reg.run_retrieved(ctx, data))
            ctx.route = res.route
        pre_ms = ctx.timings_ms.get("hook.pre", 0.0)
        await self.reg.run_post(ctx, res)  # egress redaction, HITL invariant, breaker accounting, metrics, background scoring
        res.timings_ms["hook.pre"] = pre_ms
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000  # end to end, including hooks and cache
        # only answers that passed independent verification are cached (FMEA #6); never blocked/abstained/transactional results
        if self.cache is not None and res.outcome == "ANSWERED" and res.route == "HYBRID_RAG" and not res.cache_hit:
            task = asyncio.create_task(self.cache.put(caller.tenant_id, caller.user_id, ctx.query, res.answer, verified=True))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        return res

    async def drain(self) -> None:
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
        await self.reg.drain()

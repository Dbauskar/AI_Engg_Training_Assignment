"""Subagents with enforced isolation, and the orchestrator that routes between them (docs/design.md Gates 6-7, 9).

Isolation is structural, not a prompt instruction. Each subagent talks to tools only through a ScopedHarness:
  * an allow-list of tool names (anything else is refused before the harness is even reached), and
  * a tier ceiling (the caller's tier is clamped to min(caller, ceiling)), so even a mis-configured allow-list
    cannot let a read-only subagent mutate state.

  Retrieval     read-only  hybrid_retrieve, graphify_subgraph                 (no state-changing tools)
  Execution     state-store only: submit_transaction, get_session_facts       (no retrieval, no web)
  Verification  no tools at all; independent model + deterministic guards (agents/verification.py)

Orchestrator flow for a grounded question:  triage -> retrieval -> [abstain? answer deterministically, zero LLM calls]
-> generation (bounded ReAct) -> independent verification -> answer, or abstain if the verifier keeps vetoing.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from opentelemetry import trace

from agents.react import (
    ABSTAIN_ANSWER,
    REFUSAL,
    Act,
    Evidence,
    FactMemory,
    Final,
    HarnessFactMemory,
    Message,
    ReActAgent,
    ReActConfig,
    Reasoner,
    TriageLike,
    Verifier,
    VerifyResult,
    State,
)
from app.bulkhead import Pools
from app.harness import Caller, HarnessResult, Tier, ToolExecutionHarness, make_error

_tracer = trace.get_tracer("unreliable_assistant.subagents")


# ---- isolation ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SubagentSpec:
    name: str
    allowed_tools: frozenset[str]
    max_tier: Tier


RETRIEVAL = SubagentSpec("retrieval", frozenset({"hybrid_retrieve", "graphify_subgraph"}), Tier.TIER_1_READ)
EXECUTION = SubagentSpec("execution", frozenset({"submit_transaction", "get_session_facts"}), Tier.TIER_3_MUTATE)
VERIFICATION = SubagentSpec("verification", frozenset(), Tier.TIER_1_READ)


class ScopedHarness:
    """Drop-in replacement for ToolExecutionHarness (same call/input_fields surface) limited to one subagent's powers."""

    def __init__(self, inner: ToolExecutionHarness, spec: SubagentSpec) -> None:
        self.inner, self.spec = inner, spec
        self.denials: list[str] = []

    def input_fields(self, tool_name: str) -> frozenset[str] | None:
        return self.inner.input_fields(tool_name) if tool_name in self.spec.allowed_tools else None

    async def call(self, tool_name: str, raw_args: dict[str, Any], caller: Caller) -> HarnessResult:
        if tool_name not in self.spec.allowed_tools:
            self.denials.append(tool_name)
            with _tracer.start_as_current_span("subagent.denied") as sp:  # tool NAME only, never arguments
                sp.set_attribute("subagent", self.spec.name)
                sp.set_attribute("tool.name", tool_name)
            return HarnessResult(ok=False, error=make_error(
                "E_PERMISSION_DENIED", f"The {self.spec.name} subagent may not use this tool."))
        clamped = Caller(caller.tenant_id, caller.user_id, min(caller.max_tier, self.spec.max_tier))
        return await self.inner.call(tool_name, raw_args, clamped)


# ---- query rewriting (deterministic) ------------------------------------------------------------------------------------
_FILLER = re.compile(r"^\s*(hi|hello|hey)?[\s,!.]*(please|kindly)?[\s,]*(could you|can you|would you)?\s*(please )?"
                     r"(tell me|let me know|explain|i would like to know|i'd like to know|i want to know)?[\s,:]*", re.I)
_IDS = [(re.compile(r"\bsku[\s_-]*(\d{1,3})\b", re.I), lambda m: f"SKU-{int(m.group(1)):03d}"),
        (re.compile(r"\blot[\s_-]*(\d{1,5})\b", re.I), lambda m: f"LOT-{int(m.group(1)):05d}"),
        (re.compile(r"\bacc-(\d{8})\b", re.I), lambda m: f"ACC-{m.group(1)}")]


def rewrite_query(query: str) -> str:
    """Normalise entity ids (so BM25 can match 'SKU-099') and drop conversational filler. Never returns empty."""
    q = re.sub(r"\s+", " ", query).strip()
    for rx, fn in _IDS:
        q = rx.sub(fn, q)
    stripped = _FILLER.sub("", q, count=1).strip()
    return stripped if len(stripped.split()) >= 2 else q


# ---- retrieval subagent ---------------------------------------------------------------------------------------------------
@dataclass
class RetrievalOutcome:
    ok: bool
    data: dict[str, Any] | None
    error_code: str | None
    latency_ms: float


class RetrievalSubagent:
    def __init__(self, harness: ToolExecutionHarness, score_threshold: float | None = None) -> None:
        self.scoped = ScopedHarness(harness, RETRIEVAL)
        self.score_threshold = score_threshold  # None = the tool contract's default

    async def retrieve(self, caller: Caller, query: str) -> RetrievalOutcome:
        t0 = time.perf_counter()
        args = {"tenant_id": caller.tenant_id, "user_id": caller.user_id, "request_id": str(uuid.uuid4()),
                "query": rewrite_query(query)}
        if self.score_threshold is not None:
            args["score_threshold"] = self.score_threshold
        res = await self.scoped.call("hybrid_retrieve", args, caller)
        ms = (time.perf_counter() - t0) * 1000
        return RetrievalOutcome(res.ok, res.data if res.ok else None, res.error.code if res.error else None, ms)


    async def subgraph(self, caller: Caller, query: str, hub_ids: list[str]) -> dict[str, Any] | None:
        """Compact graph context for the retrieved policies. Any failure just means 'no graph context': never an error."""
        args = {"tenant_id": caller.tenant_id, "user_id": caller.user_id, "request_id": str(uuid.uuid4()),
                "query": rewrite_query(query), "hub_ids": hub_ids[:8]}
        res = await self.scoped.call("graphify_subgraph", args, caller)
        return res.data if res.ok else None


def graph_prompt_context(graph: dict[str, Any], chunks: list[dict[str, Any]]) -> str | None:
    """Render a subgraph for the model, citing RETRIEVED CHUNK ids (so answers stay citeable and verifiable).
    Notes whose policy was not retrieved cannot be cited, so they are left out. None = nothing usable."""
    chunk_of: dict[str, str] = {}
    for c in chunks:
        chunk_of.setdefault(c["chunk_id"].split("-")[0], c["chunk_id"])
    by_cite: dict[str, list[str]] = {}
    seen: set[str] = set()
    for n in graph.get("nodes", []):
        cite = chunk_of.get(n.get("hub") or "")
        text = n["text"].split("\n")[0]
        if n["type"] == "note" and cite and text not in seen:
            seen.add(text)
            by_cite.setdefault(cite, []).append(text)  # one citation per chunk, not one per note (the id costs ~7 tokens)
    return "\n".join(f"[{cite}] " + " ".join(texts) for cite, texts in by_cite.items()) or None


# ---- verification wiring --------------------------------------------------------------------------------------------
class StateStoreAuditSink:
    """Writes verification verdicts (counts and digests only) into the tenant's hash-chained audit log."""

    def __init__(self, store: Any) -> None:
        self.store = store

    async def record(self, tenant_id: str, actor: str, event_type: str, payload: dict[str, Any]) -> None:
        await self.store.append_audit_event(tenant_id, actor, event_type, payload)


class _CountingVerifier:
    def __init__(self, inner: Verifier) -> None:
        self.inner, self.vetoes, self.checks = inner, 0, 0

    async def verify(self, final: Final, evidence: Evidence, route: str) -> VerifyResult:
        self.checks += 1
        r = await self.inner.verify(final, evidence, route)
        self.vetoes += not r.ok
        return r


class _CountingReasoner:
    def __init__(self, inner: Reasoner) -> None:
        self.inner, self.calls = inner, 0

    async def next_step(self, prompt: list[Message]) -> Act | Final:
        self.calls += 1
        return await self.inner.next_step(prompt)


class ExtractiveReasoner:
    """Deterministic, LLM-free answerer: quotes the most query-relevant sentences of the top chunk with a citation.
    Used as the FMEA #4 fallback (LLM unavailable) and as a stand-in in integration tests."""

    def __init__(self, chunks: list[dict[str, Any]], query: str, max_sentences: int = 2) -> None:
        self.chunks, self.query, self.max_sentences = chunks, query, max_sentences

    async def next_step(self, prompt: list[Message]) -> Act | Final:
        from agents.verification import _sentences
        from mcp_servers.rag_pipeline import bm25_tokens

        top = self.chunks[0]
        q = set(bm25_tokens(self.query))
        sents = _sentences(re.sub(r"^#+ .*\n+", "", top["text"].strip()))  # drop the markdown title line
        best = sorted(range(len(sents)), key=lambda i: (-len(q & set(bm25_tokens(sents[i]))), i))[: self.max_sentences]
        body = " ".join(sents[i] for i in sorted(best))
        return Final(f"{body} [{top['chunk_id']}]", citations=(top["chunk_id"],))


# ---- orchestrator ---------------------------------------------------------------------------------------------------------
@dataclass
class AnswerResult:
    answer: str
    route: str
    outcome: str  # ANSWERED | ABSTAINED | REFUSED | FROZEN | DEGRADED | HALTED
    abstained: bool = False
    reason: str | None = None
    llm_calls: int = 0
    verifier_vetoes: int = 0
    frozen_for_approval: bool = False
    graph_used: bool = False
    cache_hit: bool = False
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    halt_reason: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


RetrievalFilter = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class Orchestrator:
    def __init__(self, harness: ToolExecutionHarness, triage: TriageLike, verifier: Verifier,
                 reasoner_factory: Callable[[str, dict[str, Any] | None], Reasoner],
                 memory: FactMemory | None = None, pools: Pools | None = None,
                 config: ReActConfig | None = None, graph_mode: str = "off", retrieval_threshold: float | None = None) -> None:
        assert graph_mode in ("off", "measure", "use")  # off: no graph; measure: build + log tokens only; use: show it to the model
        self.graph_mode = graph_mode
        self.h, self.triage, self.verifier, self.reasoner_factory = harness, triage, verifier, reasoner_factory
        self.memory = memory or HarnessFactMemory(harness)
        self.pools, self.cfg = pools, config or ReActConfig()
        self.retrieval = RetrievalSubagent(harness, retrieval_threshold)

    async def _classify(self, query: str):
        if self.pools:  # triage has its own pool so it is never queued behind slow retrieval
            return await self.pools.triage.run_sync(self.triage.classify, query)
        return await asyncio.to_thread(self.triage.classify, query)

    async def answer(self, caller: Caller, query: str, session_id: str, retrieval_filter: RetrievalFilter | None = None) -> AnswerResult:
        t0 = time.perf_counter()
        with _tracer.start_as_current_span("orchestrator.answer") as span:
            span.set_attribute("tenant.id", caller.tenant_id)
            tri = await self._classify(query)
            timings = {"triage": (time.perf_counter() - t0) * 1000}
            span.set_attribute("orchestrator.route", tri.route)
            res = await self._route(caller, query, session_id, tri, timings, retrieval_filter)
            timings["total"] = (time.perf_counter() - t0) * 1000
            res.timings_ms = timings
            span.set_attribute("orchestrator.outcome", res.outcome)
            span.set_attribute("orchestrator.llm_calls", res.llm_calls)
            return res

    async def _route(self, caller, query, session_id, tri, timings, retrieval_filter=None) -> AnswerResult:
        if tri.route == "REJECT_OOS":
            return AnswerResult(REFUSAL, tri.route, "REFUSED", trace=["TRIAGE", "DONE"])

        if tri.route == "TRANSACTION_EXEC":  # Execution subagent: state-store tools only; Tier 3 freezes at PENDING_APPROVAL
            reasoner = _CountingReasoner(self.reasoner_factory(query, None))
            agent = ReActAgent(ScopedHarness(self.h, EXECUTION), self.triage, reasoner, self.verifier, self.memory,
                               self.cfg, llm_pool=self.pools.llm if self.pools else None)
            r = await agent.run(caller, query, session_id, triage_result=tri)
            # DONE without a freeze = the agent asked for missing details (nothing was submitted or executed)
            outcome = "FROZEN" if r.frozen_for_approval else ("NEEDS_INFO" if r.state is State.DONE else "HALTED")
            return AnswerResult(r.answer, tri.route, outcome, llm_calls=reasoner.calls,
                                frozen_for_approval=r.frozen_for_approval, halt_reason=r.halt_reason, trace=r.trace)

        if tri.route == "DIRECT_ANSWER":
            reasoner = _CountingReasoner(self.reasoner_factory(query, None))
            agent = ReActAgent(ScopedHarness(self.h, SubagentSpec("direct", frozenset(), Tier.TIER_1_READ)), self.triage,
                               reasoner, self.verifier, self.memory, self.cfg, llm_pool=self.pools.llm if self.pools else None)
            r = await agent.run(caller, query, session_id, triage_result=tri)
            return AnswerResult(r.answer, tri.route, "ANSWERED" if r.state is State.DONE else "HALTED",
                                llm_calls=reasoner.calls, halt_reason=r.halt_reason, trace=r.trace)

        # ---- HYBRID_RAG ----
        out = await self.retrieval.retrieve(caller, query)
        t_retrieved = time.perf_counter()
        timings["retrieval"] = out.latency_ms
        if not out.ok:  # FMEA #3: no retrieval -> say so; never improvise an answer
            return AnswerResult("The knowledge base is temporarily unavailable, so I can't answer this reliably right now.",
                                tri.route, "DEGRADED", reason=out.error_code, trace=["TRIAGE", "RETRIEVAL_FAILED"])
        assert out.data is not None
        if self.graph_mode != "off" and out.data.get("chunks"):
            hubs = sorted({c["chunk_id"].split("-")[0] for c in out.data["chunks"]})
            out.data["graph_context"] = await self.retrieval.subgraph(caller, query, hubs)
        if retrieval_filter is not None:  # e.g. quarantine poisoned chunks before the model can read them
            out.data = await retrieval_filter(out.data)
        graph_used = False
        if self.graph_mode == "use" and out.data.get("graph_context"):
            shown = graph_prompt_context(out.data["graph_context"], out.data["chunks"])
            if shown:  # otherwise fall back to the normal chunk context
                out.data = {**out.data, "prompt_context": shown}
                graph_used = True
        if out.data.get("abstain"):  # deterministic short-circuit: no LLM, no tokens, nothing to hallucinate
            with _tracer.start_as_current_span("orchestrator.abstain") as sp:
                sp.set_attribute("abstain.top_score", float(out.data.get("top_score", 0.0)))
                res = AnswerResult(ABSTAIN_ANSWER, tri.route, "ABSTAINED", abstained=True, reason="below_threshold",
                                   llm_calls=0, trace=["TRIAGE", "RETRIEVAL", "ABSTAIN"])
            timings["abstain_overhead"] = (time.perf_counter() - t_retrieved) * 1000  # decision time after retrieval
            return res

        reasoner = _CountingReasoner(self.reasoner_factory(query, out.data))
        verifier = _CountingVerifier(self.verifier)
        agent = ReActAgent(ScopedHarness(self.h, RETRIEVAL), self.triage, reasoner, verifier, self.memory, self.cfg,
                           llm_pool=self.pools.llm if self.pools else None)
        t_gen = time.perf_counter()
        r = await agent.run(caller, query, session_id, triage_result=tri, seed_context=out.data)
        timings["generation"] = (time.perf_counter() - t_gen) * 1000
        if r.state is State.DONE:
            ids = [c["chunk_id"] for c in out.data.get("chunks", [])]
            return AnswerResult(r.answer, tri.route, "ANSWERED", llm_calls=reasoner.calls, graph_used=graph_used,
                                verifier_vetoes=verifier.vetoes, trace=r.trace, retrieved_chunk_ids=ids,
                                cited_chunk_ids=[i for i in ids if i in r.answer])
        if verifier.vetoes:  # every draft was vetoed: the safe answer is an abstention, never the vetoed text
            return AnswerResult(ABSTAIN_ANSWER, tri.route, "ABSTAINED", abstained=True, reason="verification_vetoed",
                                llm_calls=reasoner.calls, verifier_vetoes=verifier.vetoes, halt_reason=r.halt_reason, trace=r.trace)
        return AnswerResult(r.answer, tri.route, "HALTED", llm_calls=reasoner.calls, halt_reason=r.halt_reason, trace=r.trace)

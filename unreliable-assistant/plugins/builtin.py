"""Built-in lifecycle plugins: pii_redactor, injection_shield, circuit_breaker, otel_exporter, hitl_gate.

Every plugin records only counts, ids and error classes in ctx.events, never user text (CLAUDE.md rule 4).
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable

from opentelemetry import metrics

from app.injection import InjectionDetector
from app.pii import PiiDetector, shared_detector
from app.tokens import count_tokens
from plugins.registry import Block, RequestContext, hookimpl


class PiiRedactor:
    """PRE: replaces PII in the query with <TYPE> tokens before anything else sees it. POST: egress scan on the answer
    (FMEA #10: a second pass catches what the generator might have reproduced or invented)."""

    def __init__(self, detector: PiiDetector | None = None) -> None:
        self.det = detector or shared_detector()

    @hookimpl
    def pre_execute(self, ctx: RequestContext) -> None:
        r = self.det.redact(ctx.query)
        ctx.query = r.text
        if r.counts:
            ctx.pii_counts = dict(r.counts)
            ctx.event("pii_redactor", "pii.redacted", counts=r.counts)

    @hookimpl
    def post_execute(self, ctx: RequestContext, result: Any) -> None:
        answer = getattr(result, "answer", None)
        if answer:
            r = self.det.redact(answer)
            if r.counts:
                result.answer = r.text
                ctx.event("pii_redactor", "pii.egress_redacted", counts=r.counts)


class InjectionShield:
    """PRE: blocks direct injection in the query. scan_retrieved: quarantines poisoned chunks (indirect injection)
    so the model never reads them; if every chunk is quarantined the request abstains."""

    BLOCK_MESSAGE = "I can't process that request."

    def __init__(self, detector: InjectionDetector | None = None) -> None:
        self.det = detector or InjectionDetector()

    @hookimpl
    def pre_execute(self, ctx: RequestContext) -> None:
        v = self.det.scan(ctx.query)
        if v.is_attack:
            ctx.event("injection_shield", "injection.blocked", family=v.family, layer=v.layer, source="query")
            raise Block("E_INJECTION_BLOCKED", self.BLOCK_MESSAGE, "prompt_injection")

    @hookimpl
    def scan_retrieved(self, ctx: RequestContext, data: dict) -> dict:
        keep, dropped = [], []
        for c in data.get("chunks", []):
            v = self.det.scan(str(c.get("text", "")))
            (dropped if v.is_attack else keep).append((c, v))
        for c, v in dropped:
            ctx.event("injection_shield", "injection.chunk_quarantined", chunk_id=c.get("chunk_id"), family=v.family, layer=v.layer)
        gc = data.get("graph_context")
        if gc:  # graph context is derived from the same documents: scan it too, and drop it wholesale if anything is poisoned
            bad = [n for n in gc.get("nodes", []) if self.det.scan(str(n.get("text", ""))).is_attack]
            if bad or dropped:
                ctx.event("injection_shield", "injection.graph_context_dropped", nodes=[n.get("id") for n in bad], with_chunks=bool(dropped))
                data = {**data, "graph_context": None}
        if not dropped:
            return data
        out = {**data, "chunks": [c for c, _ in keep], "graph_context": None if gc else data.get("graph_context")}
        if not keep:
            out["abstain"] = True  # nothing trustworthy left to answer from
        return out


class CircuitBreaker:
    """Session-level breaker. PRE: refuses a session that recently tripped the loop breaker repeatedly. POST: records
    BREAKER halts. (The per-tool breaker inside the ReAct loop is separate and finer-grained.) In-process state:
    FMEA #1 says the breaker degrades to local counters when Redis is unavailable."""

    def __init__(self, threshold: int = 2, window_s: float = 120.0, cooldown_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold, self.window_s, self.cooldown_s, self.clock = threshold, window_s, cooldown_s, clock
        self._fails: dict[tuple[str, str], list[float]] = {}
        self._open_until: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    @hookimpl
    def pre_execute(self, ctx: RequestContext) -> None:
        key = (ctx.tenant_id, ctx.session_id)
        with self._lock:
            until = self._open_until.get(key, 0.0)
        if self.clock() < until:
            raise Block("E_SESSION_PAUSED", "This conversation is paused after repeated failures. Please try again shortly.", "circuit_breaker")

    @hookimpl
    def post_execute(self, ctx: RequestContext, result: Any) -> None:
        key, now = (ctx.tenant_id, ctx.session_id), self.clock()
        with self._lock:
            if getattr(result, "halt_reason", None) == "BREAKER":
                recent = [t for t in self._fails.get(key, []) if now - t < self.window_s] + [now]
                self._fails[key] = recent
                if len(recent) >= self.threshold:
                    self._open_until[key] = now + self.cooldown_s
                    self._fails[key] = []
                    ctx.event("circuit_breaker", "breaker.opened", cooldown_s=self.cooldown_s)
            elif getattr(result, "outcome", None) in ("ANSWERED", "ABSTAINED", "FROZEN", "REFUSED"):
                self._fails.pop(key, None)  # a healthy request resets the streak


class OtelExporter:
    """POST: request counters/latency as OpenTelemetry metrics. Attributes are low-cardinality and PII-free."""

    def __init__(self, meter: Any = None) -> None:
        m = meter or metrics.get_meter("unreliable_assistant")
        self.requests = m.create_counter("assistant.requests")
        self.latency = m.create_histogram("assistant.request.latency_ms")
        self.pii = m.create_counter("assistant.pii.redacted")
        self.blocked = m.create_counter("assistant.blocked")

    @hookimpl
    def post_execute(self, ctx: RequestContext, result: Any) -> None:
        route, outcome = str(getattr(result, "route", "?")), str(getattr(result, "outcome", "?"))
        self.requests.add(1, {"route": route, "outcome": outcome})
        # measured from the request's own start, so cache hits and blocked requests are included (they have no orchestrator timings)
        self.latency.record((time.perf_counter() - ctx.started) * 1000, {"route": route})
        for kind, n in ctx.pii_counts.items():
            self.pii.add(n, {"type": kind})
        if outcome == "BLOCKED" or ctx.blocked is not None:
            self.blocked.add(1, {"category": ctx.blocked.category if ctx.blocked else "unknown"})


class HitlGate:
    """POST invariant check (design invariant B1): nothing may reach the user claiming a state-changing action was
    executed. A transaction request must end FROZEN at PENDING_APPROVAL; any answer on any route that claims a transfer,
    payment or order was carried out is vetoed. Fail-closed."""

    SAFE = "This request needs human approval and has not been executed."
    EXECUTED = re.compile(
        r"\b(?:i|we)\s+(?:have\s+|'ve\s+)?(?:executed|completed|processed|submitted|transferred|sent|paid|wired|placed)\b[^.\n]{0,40}"
        r"\b(?:transfer|payment|order|wire|transaction|funds|money)\b"
        r"|\b(?:i|we)\s+(?:have\s+|'ve\s+)?(?:transferred|sent|paid|wired|remitted)\b[^.\n]{0,25}\d"
        r"|\b(?:transfer|payment|order|wire|transaction)\b[^.\n]{0,30}\b(?:has|have)\s+been\s+(?:executed|completed|processed|sent|made|placed)\b", re.I)
    OK_TXN_OUTCOMES = {"FROZEN", "NEEDS_INFO", "REFUSED", "HALTED", "DEGRADED", "BLOCKED"}

    @hookimpl
    def post_execute(self, ctx: RequestContext, result: Any) -> None:
        route, outcome = getattr(result, "route", None), getattr(result, "outcome", None)
        reason = None
        if route == "TRANSACTION_EXEC":
            if outcome not in self.OK_TXN_OUTCOMES:
                reason = "transaction_not_frozen"
            elif outcome == "FROZEN" and not getattr(result, "frozen_for_approval", False):
                reason = "frozen_flag_missing"
        if reason is None and outcome != "FROZEN" and self.EXECUTED.search(str(getattr(result, "answer", ""))):
            reason = "claims_execution"
        if reason:
            ctx.event("hitl_gate", "hitl.violation", reason=reason, route=route, outcome=outcome)
            raise Block("E_HITL_VIOLATION", self.SAFE, "hitl")


class TokenCounter:
    """scan_retrieved hook: logs how many tokens the retrieved context costs BEFORE (raw chunk text) and AFTER (Graphify
    subgraph, when one was built) as an audit-friendly event and OTel histograms. It never changes the data."""

    def __init__(self, meter: Any = None) -> None:
        m = meter or metrics.get_meter("unreliable_assistant")
        self.before = m.create_histogram("assistant.context.tokens_before")
        self.after = m.create_histogram("assistant.context.tokens_after")

    @hookimpl
    def scan_retrieved(self, ctx: RequestContext, data: dict) -> dict:
        before = count_tokens("\n".join(str(c.get("text", "")) for c in data.get("chunks", [])))
        gc = data.get("graph_context")
        after = int(gc["tokens"]) if gc else None
        ctx.flags["context_tokens"] = {"before": before, "after": after}
        ctx.event("token_counter", "tokens.context", before=before, after=after,
                  reduction=round(1 - after / before, 3) if after is not None and before else None)
        self.before.record(before)
        if after is not None:
            self.after.record(after)
        return data


class RagasScorer:
    """POST, background: ONLINE quality proxies recorded as metrics under the `hook.post.ragas` span.

    These are cheap stand-ins, NOT the RAGAS library metrics (those need an LLM judge and a golden set; they are computed
    offline in Prompt 11 for KPI-20..22):
      context_utilization    cited chunks / retrieved chunks
      answer_relevancy_proxy cosine(query, answer) under the shared embedder
    Runs after the response is ready, so it never adds latency; it only sees redacted text."""

    def __init__(self, embedder: Any = None, meter: Any = None) -> None:
        m = meter or metrics.get_meter("unreliable_assistant")
        self.util = m.create_histogram("assistant.quality.context_utilization")
        self.rel = m.create_histogram("assistant.quality.answer_relevancy_proxy")
        self.embedder = embedder

    @hookimpl
    def post_execute(self, ctx: RequestContext, result: Any) -> None:
        import numpy as np
        from opentelemetry import trace

        with trace.get_tracer("unreliable_assistant.ragas").start_as_current_span("hook.post.ragas") as span:
            if getattr(result, "route", None) != "HYBRID_RAG" or getattr(result, "outcome", None) != "ANSWERED":
                span.set_attribute("ragas.skipped", True)
                return
            n_ret, n_cit = len(result.retrieved_chunk_ids), len(result.cited_chunk_ids)
            if n_ret:
                util = n_cit / n_ret
                span.set_attribute("ragas.context_utilization", util)
                self.util.record(util)
            if self.embedder is not None:
                v = np.asarray(self.embedder.encode([ctx.query, result.answer], False), dtype=np.float32)
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
                rel = float(v[0] @ v[1])
                span.set_attribute("ragas.answer_relevancy_proxy", rel)
                self.rel.record(rel)

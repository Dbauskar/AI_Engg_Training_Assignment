"""Logic behind the six-tab Streamlit workbench (app.py is a thin UI over this module).

Runs the REAL pipeline on the in-memory local stack (app/localstack.py): hooks, triage, hybrid RAG, verification and the
tool harness are real; Postgres and Redis are stand-ins. Nothing here is a mock of the decision logic.
Raw PII is never displayed or stored: user text goes through the pipeline's redactor first, and the probe tab shows only
verdicts/codes, never the payload back (CLAUDE.md rule 4).
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from opentelemetry import trace
from rank_bm25 import BM25Okapi

from agents.react import CircuitBreaker
from app import kpi
from app.auth import mint_token
from app.harness import Caller, Tier
from app.localstack import build_local_runtime
from app.runtime import Runtime, Settings
from app.tokens import count_tokens
from mcp_servers.rag_pipeline import POOL, TOP_K, bm25_tokens, rrf_fuse
from plugins.registry import RequestContext

ROOT = Path(__file__).resolve().parents[1]
_tracer = trace.get_tracer("unreliable_assistant.workbench")
TENANT = "acme-corp"
SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.M)


# ---------------------------------------------------------------- tab 1: architecture
def design_checklist(path: Path = ROOT / "docs" / "design.md") -> dict[str, Any]:
    with _tracer.start_as_current_span("workbench.design_checklist"):
        text = path.read_text()
        locked = text.lstrip().startswith("STATUS: LOCKED")
        found = {int(n): t.strip() for n, t in SECTION_RE.findall(text)}
        want = {1: "charter", 2: "tool contracts", 3: "topology", 4: "latency", 5: "kpi", 6: "fmea", 7: "adr"}
        items = [{"section": n, "title": found.get(n, "(missing)"), "present": n in found} for n in want]
        return {"locked": locked, "items": items, "all_present": all(i["present"] for i in items),
                "diagram": str(ROOT / "docs" / "workflow_orchestration.png")}


class Bench:
    """Owns an event loop in a background thread and one local runtime (Streamlit reruns the script on every click)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="workbench-loop").start()
        self.settings = settings or Settings.from_env()
        self.rt: Runtime = self.run(self._boot())
        self.breaker = CircuitBreaker()
        self.session_id = str(uuid.uuid4())

    def run(self, coro: Any, timeout: float = 120) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _boot(self) -> Runtime:
        rt = await build_local_runtime(self.settings)
        await rt.warmup()
        return rt

    # ---- live triage console + answer + ReAct trace
    def triage(self, query: str) -> dict[str, Any]:
        with _tracer.start_as_current_span("workbench.triage"):
            ctx = RequestContext.create(TENANT, "wb", self.session_id, query)
            self.run(self.rt.registry.run_pre(ctx))  # redact first: the classifier never sees raw PII
            if ctx.blocked is not None:
                return {"blocked": ctx.blocked.code, "message": ctx.blocked.public_message}
            r = self.rt.pipeline.orch.triage.classify(ctx.query)
            return {"intent": r.intent, "domain": r.domain, "risk_tier": r.risk_tier, "complexity": r.complexity,
                    "route": str(getattr(r.route, "value", r.route)), "confidence": round(r.confidence, 3),
                    "source": r.source, "latency_ms": round(r.latency_ms, 1)}

    def answer(self, query: str, user: str = "wb-user", tier: Tier = Tier.TIER_3_MUTATE, session_id: str | None = None) -> dict[str, Any]:
        with _tracer.start_as_current_span("workbench.answer"):
            res = self.run(self.rt.pipeline.answer(Caller(TENANT, user, tier), query, session_id or self.session_id))
            return {"answer": res.answer, "route": res.route, "outcome": res.outcome, "abstained": res.abstained,
                    "reason": res.reason, "cache_hit": res.cache_hit, "frozen": res.frozen_for_approval,
                    "trace": list(res.trace), "cited": list(res.cited_chunk_ids),
                    "timings_ms": {k: round(v, 1) for k, v in res.timings_ms.items()}}

    # ---- HITL queue (separation of duties enforced by the store: requester cannot approve own request)
    def pending(self) -> list[dict[str, Any]]:
        return self.run(self.rt.store.pending(TENANT))

    def decide(self, approval_id: str, approve: bool, approver: str) -> dict[str, Any]:
        with _tracer.start_as_current_span("workbench.hitl_decide") as sp:
            sp.set_attribute("hitl.approve", approve)
            try:
                return {"ok": True, "row": self.run(self.rt.store.decide(TENANT, approval_id, approve, approver))}
            except (PermissionError, KeyError, ValueError) as e:
                return {"ok": False, "error": str(e)}

    def audit_ok(self) -> tuple[bool, int | None, int]:
        return self.run(self.rt.store.verify_chain(TENANT))

    # ---- tab 2: RAG lab
    def rag_lab(self, query: str, k1: float, b: float, rrf_k: int, threshold: float) -> dict[str, Any]:
        """Side-by-side stage scores with the BM25 and RRF parameters, and the abstain threshold, under the caller's control."""
        with _tracer.start_as_current_span("workbench.rag_lab"):
            ret = self.rt.pipeline.orch.retriever if hasattr(self.rt.pipeline.orch, "retriever") else self._retriever()
            ti = ret.index_for(TENANT)
            bm = self._bm25(ti, round(k1, 2), round(b, 2))
            sc = bm.get_scores(bm25_tokens(query))
            bm_ids = [int(i) for i in np.argsort(-sc)[:TOP_K] if sc[i] > 0]
            q = ret.embedder.encode([query], is_query=True)
            ds, di = ti.faiss_index.search(q, min(TOP_K, len(ti.chunks)))
            de_ids, de_sc = [int(i) for i in di[0] if i >= 0], {int(i): float(s) for i, s in zip(di[0], ds[0]) if i >= 0}
            pool = rrf_fuse([bm_ids, de_ids], [c.text_hash for c in ti.chunks], int(rrf_k), POOL)
            rr = ret.reranker.score(query, [ti.chunks[i].text for i in pool])
            rr_by = {i: float(s) for i, s in zip(pool, rr)}
            rrf_score = {i: sum(1 / (rrf_k + r) for lst in (bm_ids, de_ids) for r, j in enumerate(lst, 1) if j == i) for i in pool}
            name = lambda i: ti.chunks[i].chunk_id  # noqa: E731
            ranked = sorted(pool, key=lambda i: -rr_by[i])
            kept = [i for i in ranked if rr_by[i] > threshold][:4]
            return {
                "bm25": [{"chunk": name(i), "score": round(float(sc[i]), 3)} for i in bm_ids[:8]],
                "dense": [{"chunk": name(i), "score": round(de_sc[i], 3)} for i in de_ids[:8]],
                "rrf": [{"chunk": name(i), "score": round(rrf_score[i], 5)} for i in pool[:8]],
                "rerank": [{"chunk": name(i), "score": round(rr_by[i], 3), "kept": i in kept} for i in ranked[:8]],
                "abstain": not kept, "top_score": round(rr_by[ranked[0]], 3) if ranked else 0.0,
            }

    _bm_cache: dict[tuple[int, float, float], BM25Okapi] = {}

    def _retriever(self) -> Any:
        if not hasattr(self, "_ret"):
            from mcp_servers.rag_pipeline import HybridRetriever
            self._ret = HybridRetriever(self.settings.index_root, self.settings.embed_dir, self.settings.rerank_dir,
                                        embedder=self.rt.pipeline.orch.triage.embedder if hasattr(self.rt.pipeline.orch.triage, "embedder") else None)
        return self._ret

    def _bm25(self, ti: Any, k1: float, b: float) -> BM25Okapi:
        key = (id(ti), k1, b)
        if key not in self._bm_cache:
            self._bm_cache[key] = BM25Okapi([bm25_tokens(c.text) for c in ti.chunks], k1=k1, b=b)
        return self._bm_cache[key]

    # ---- tab 3: MCP & plugin inspector
    def servers(self) -> list[dict[str, Any]]:
        from mcp_servers.manifest import load_verified_config
        out = []
        cfg = json.loads((ROOT / "mcp_config.json").read_text())["mcpServers"]
        try:
            verified = load_verified_config(ROOT / "mcp_config.json", ROOT)
        except Exception as e:  # noqa: BLE001 - show the failure; a bad manifest must be visible, not hidden
            verified, err = {}, type(e).__name__
        else:
            err = None
        for name, c in cfg.items():
            out.append({"server": name, "transport": c.get("transport"), "required": c.get("required", False),
                        "manifest_verified": name in verified, "error": err})
        return out

    @staticmethod
    def jsonrpc_example(query: str) -> dict[str, Any]:
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "hybrid_retrieve", "arguments": {
            "tenant_id": TENANT, "user_id": "wb-user", "request_id": str(uuid.uuid4()), "query": query, "top_k": 4}}}
        return {"request": req, "note": "illustrative envelope; the live call goes through ToolExecutionHarness in-process"}

    def plugins(self) -> list[dict[str, Any]]:
        return self.rt.registry.status()

    def toggle_plugin(self, name: str, enabled: bool) -> str | None:
        try:
            self.rt.registry.set_enabled(name, enabled, "workbench")
            return None
        except Exception as e:  # noqa: BLE001 - e.g. a required fail-closed plugin cannot be disabled: surface the reason
            return str(e)

    # ---- tab 4: memory & graphify tokens
    def graph_tokens(self, query: str) -> dict[str, Any]:
        from mcp_servers.graphify import Graph
        g = Graph.load(ROOT / "data" / "graph" / TENANT / "graph.json", self.rt.pipeline.orch.triage.embedder
                       if hasattr(self.rt.pipeline.orch.triage, "embedder") else None)
        sub = g.subgraph(query, depth=2)
        ret = self._retriever().retrieve(TENANT, query)
        naive = sum(count_tokens(c.text) for c in ret.chunks)
        return {"anchor": sub.anchor, "nodes": [{"id": n.id, "type": n.type, "distance": n.distance} for n in sub.nodes],
                "graph_tokens": sub.tokens, "naive_top4_tokens": naive,
                "reduction": (1 - sub.tokens / naive) if naive else None, "rendered": sub.rendered}

    def facts(self, session_id: str | None = None) -> list[dict[str, Any]]:
        sid = session_id or self.session_id
        return [{"fact_key": k[2], "value": f.value, "verified": f.verified}
                for k, f in self.rt.store.facts.items() if k[0] == TENANT and k[1] == sid]

    # ---- tab 5: red team & circuit breaker
    def probe(self, text: str) -> dict[str, Any]:
        """Run text through the real pre-hooks. Returns the verdict only, never the payload."""
        with _tracer.start_as_current_span("workbench.probe") as sp:
            ctx = RequestContext.create(TENANT, "probe", self.session_id, text)
            self.run(self.rt.registry.run_pre(ctx))
            blocked = ctx.blocked is not None
            sp.set_attribute("probe.blocked", blocked)
            return {"blocked": blocked, "code": ctx.blocked.code if blocked else None,
                    "text_changed_by_redaction": (ctx.query != text) and not blocked,
                    "hook_pre_ms": round(ctx.timings_ms.get("hook.pre", 0.0), 1)}

    def breaker_fail(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        key = self.breaker.key(tool, args)
        tripped = self.breaker.record_failure(key)
        return {"key": f"{key[0]}:{key[1][:10]}", "tripped": tripped, "open": self.breaker.is_open(key)}

    def breaker_reset(self) -> None:
        self.breaker = CircuitBreaker()

    # ---- tab 6: scorecard
    @staticmethod
    def scorecard() -> dict[str, Any]:
        rows = kpi.evaluate()
        return {"rows": kpi.as_dicts(rows), "summary": kpi.summary(rows)}


def dev_token(user: str = "wb-user", tier: Tier = Tier.TIER_3_MUTATE) -> str:
    s = Settings.from_env()
    return mint_token(s.token_secret, user, TENANT, tier, 3600)

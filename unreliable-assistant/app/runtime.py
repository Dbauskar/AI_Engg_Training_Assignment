"""Assembles the whole system: settings, models, stores, tools, plugins, cache, orchestrator and pipeline.

The tool handlers (retrieval, graph, state) run IN-PROCESS here, via the same registrar functions the standalone MCP servers
use, so there is one implementation and no MCP hop. (The signed-manifest servers remain available for out-of-process use;
this runtime does not connect to them, so KPI-30 'MCP serialization overhead' is not exercised by the API.)
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.local_reasoners import GreetingReasoner, RuleBasedExecutionReasoner
from agents.llm_reasoner import LLMReasoner, OpenRouterClient, ResilientReasoner
from agents.nli import DEFAULT_NLI_DIR, OnnxNli
from agents.pipeline import Pipeline
from agents.react import HarnessFactMemory, ReActConfig
from agents.subagents import ExtractiveReasoner, Orchestrator, StateStoreAuditSink
from agents.triage import TriageAgent
from agents.verification import VerificationSubagent
from app.bulkhead import Pools
from app.cache import SemanticCache
from app.config import load_env, pg_conninfo, redis_url
from app.harness import RedisIdempotencyStore, ToolExecutionHarness
from app.injection import CLF_PATH, LEX_PATH, EmbeddingClassifier, InjectionDetector, LexicalClassifier
from app.injection_examples import ATTACK, BENIGN
from app.pii import shared_detector
from app.state_store import StateStore
from mcp_servers.graphify import Graph  # noqa: F401  (re-exported for convenience)
from mcp_servers.graphify_server import GraphStore, register_graphify_tools
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, DEFAULT_RERANK_DIR, HybridRetriever, OnnxEmbedder
from mcp_servers.rag_server import register_rag_tools
from mcp_servers.state_server import register_state_tools
from plugins import PluginRegistry, default_factories


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    token_secret: bytes
    index_root: str = "data/index"
    graph_root: str = "data/graph"
    embed_dir: str = DEFAULT_EMBED_DIR
    rerank_dir: str = DEFAULT_RERANK_DIR
    nli_dir: str = DEFAULT_NLI_DIR
    plugin_config: str = "mcp_config.json"
    graph_mode: str = "off"              # off | measure | use  (see agents/subagents.py)
    injection_semantic: bool = False     # +15-20 ms on every request; off by default (KPI-01 budget is 15 ms)
    cache_enabled: bool = True
    otlp_endpoint: str | None = None
    warmup_tenant: str | None = "acme-corp"
    max_body_bytes: int = 16_384
    service_name: str = "unreliable-assistant"

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        secret = os.environ.get("API_TOKEN_SECRET", "")
        if len(secret) < 32:
            raise RuntimeError("API_TOKEN_SECRET must be set to at least 32 characters (see .env.example)")
        e = os.environ.get
        return cls(secret.encode(), e("INDEX_ROOT", "data/index"), e("GRAPH_ROOT", "data/graph"), graph_mode=e("GRAPH_MODE", "off"),
                   injection_semantic=_flag("INJECTION_SEMANTIC", "0"), cache_enabled=_flag("CACHE_ENABLED", "1"),
                   otlp_endpoint=e("OTEL_EXPORTER_OTLP_ENDPOINT") or None, warmup_tenant=e("WARMUP_TENANT", "acme-corp") or None)


@dataclass
class Runtime:
    settings: Settings
    pools: Pools
    pipeline: Pipeline
    registry: PluginRegistry
    cache: SemanticCache | None
    harness: ToolExecutionHarness
    store: Any = None
    redis: Any = None
    probes: dict[str, Any] = field(default_factory=dict)   # name -> async callable returning True when healthy
    warmups: list[Any] = field(default_factory=list)       # async callables run once at startup
    ready: bool = False
    warm_seconds: float | None = None
    llm: Any = None   # OpenRouterClient or None (extractive answerer only)
    judge_client: Any = None   # OpenRouterClient behind the second-stage claim verifier, or None

    async def warmup(self) -> None:
        """Pay the first-request penalty (ONNX shape planning, spaCy load, connection setup) before taking traffic."""
        import time

        t0 = time.perf_counter()
        for w in self.warmups:
            try:
                await w()
            except Exception:  # a failed warm-up must not stop the service: readiness probes report real dependency state
                pass
        self.warm_seconds = time.perf_counter() - t0
        self.ready = True

    async def dependencies(self) -> dict[str, bool]:
        out = {}
        for name, probe in self.probes.items():
            try:
                out[name] = bool(await asyncio.wait_for(probe(), 1.0))
            except Exception:
                out[name] = False
        return out

    async def close(self) -> None:
        await self.pipeline.drain()
        if self.store is not None:
            await self.store.close()
        if self.redis is not None:
            await self.redis.aclose()
        if self.llm is not None:
            await self.llm.close()
        if self.judge_client is not None:
            await self.judge_client.close()
        self.pools.shutdown()


async def build_runtime(s: Settings) -> Runtime:
    import redis.asyncio as aioredis

    pools = Pools.from_env()
    embedder = OnnxEmbedder(s.embed_dir, threads=2)  # ONE embedder session shared by retrieval, triage, graph anchoring, cache, scoring
    retriever = HybridRetriever(s.index_root, s.embed_dir, s.rerank_dir, embedder=embedder)
    triage = TriageAgent(embedder)
    nli = OnnxNli(s.nli_dir)
    redis = aioredis.from_url(redis_url(), socket_timeout=2)
    store = await StateStore.create(pg_conninfo(), max_size=int(os.environ.get("DB_POOL", "20")))
    audit = StateStoreAuditSink(store)

    harness = ToolExecutionHarness(RedisIdempotencyStore(redis))
    register_rag_tools(harness, retriever, pools)
    register_state_tools(harness, store, pools)
    register_graphify_tools(harness, GraphStore(s.graph_root, embedder), pools)

    inj = InjectionDetector(embedder, ATTACK, BENIGN) if s.injection_semantic else InjectionDetector()
    if _flag("INJECTION_CLASSIFIER", "1") and CLF_PATH.exists():  # trained paraphrase detector on the SHARED embedder (memoised)
        inj._clf = EmbeddingClassifier(embedder)
        if LEX_PATH.exists():  # TF-IDF paraphrase detector: no encoder call, ~0.3 ms
            inj._lex = LexicalClassifier()
    if _flag("INJECTION_LLM", "1") and os.environ.get("OPENROUTER_API_KEY") and _flag("LLM_ENABLED", "1"):  # borderline-band second opinion
        from agents.llm_verifier import LLMInjectionJudge
        inj._esc = LLMInjectionJudge(os.environ["OPENROUTER_API_KEY"], os.environ.get("INJECTION_MODEL", "openai/gpt-6-luna"))
    registry = PluginRegistry.from_config(s.plugin_config, default_factories(shared_detector(), inj, embedder=embedder),
                                          audit_sink=audit, runner=pools.triage.run_sync)
    llm = OpenRouterClient() if (_flag("LLM_ENABLED", "1") and os.environ.get("OPENROUTER_API_KEY")) else None
    judge = judge_client = None
    if llm is not None and _flag("LLM_VERIFIER", "1"):  # second-stage claim check by a DIFFERENT model family than the answerer
        from agents.llm_verifier import LLMClaimJudge
        judge_client = OpenRouterClient(model=os.environ.get("VERIFIER_MODEL", "openai/gpt-6-luna"), timeout_s=8.0)
        judge = LLMClaimJudge(judge_client)
    verifier = VerificationSubagent(nli, audit, pools, llm_judge=judge)

    def reasoner_for(query: str, data: dict | None):
        if data is not None:
            fallback = ExtractiveReasoner(data["chunks"], query)   # also the whole answerer when no LLM is configured
            if llm is None:
                return fallback
            return ResilientReasoner(LLMReasoner(llm, data["chunks"], query), fallback, llm.usage)
        return RuleBasedExecutionReasoner(query) if _looks_transactional(query) else GreetingReasoner()

    # The design's 120 ms step / 450 ms total budgets assume a local, fast generator. A hosted LLM cannot answer in 120 ms, so
    # with one configured the loop budgets are widened (env-tunable). This is a KNOWN conflict with KPI-31..33; see the README.
    react_cfg = ReActConfig(per_step_timeout_s=float(os.environ.get("LLM_STEP_TIMEOUT_S", "10")),
                            total_budget_s=float(os.environ.get("LLM_TOTAL_BUDGET_S", "15"))) if llm is not None else None
    orchestrator = Orchestrator(harness, triage, verifier, reasoner_for, memory=HarnessFactMemory(harness), pools=pools, config=react_cfg,
                                graph_mode=s.graph_mode)
    cache = SemanticCache(redis, lambda t: embedder.encode([t], False)[0], enabled=s.cache_enabled) if s.cache_enabled else None
    pipeline = Pipeline(orchestrator, registry, cache)

    async def ping_pg() -> bool:
        async with store.pool.connection() as c:
            await c.execute("SELECT 1")
        return True

    rt = Runtime(s, pools, pipeline, registry, cache, harness, store, redis, probes={"postgres": ping_pg, "redis": redis.ping})
    rt.llm = llm
    rt.judge_client = judge_client

    async def warm() -> None:
        await asyncio.to_thread(triage.classify, "warm up question about policy")
        await asyncio.to_thread(shared_detector().redact, "warm up Jane Doe")
        await asyncio.to_thread(nli.probs, ["Wires need two approvers."], ["Wires need approvers."])
        if s.warmup_tenant:
            await pools.rag.run_sync(retriever.retrieve, s.warmup_tenant, "warm up question about wire approval")

    rt.warmups.append(warm)
    return rt


def _looks_transactional(query: str) -> bool:
    import re
    return bool(re.search(r"\b(transfer|wire|send|pay|order|purchase|cancel|close|suspend|status|move|remit)\b", query, re.I))

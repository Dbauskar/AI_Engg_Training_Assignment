"""mcp-hybrid-rag: MCP server (stdio) exposing the 3-stage hybrid retrieval pipeline.

Tool      hybrid_retrieve   (TIER_1_READ)  - input schema is generated from app.contracts.HybridRetrieveIn
Resource  rag://chunks/{tenant_id}/{chunk_id}  - tenant is part of the URI so isolation is enforceable
                                                 (design.md v1.2 change; v1.1 had rag://chunks/{id})

Every tool call is routed through ToolExecutionHarness (CLAUDE.md rule 2): tenant guard, validation,
tier check, idempotency. The server trusts its stdio transport; authenticating the caller is the
MCP host's job.

Run:  RAG_INDEX_ROOT=data/index uv run python mcp_servers/rag_server.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python mcp_servers/rag_server.py`

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ResourceError  # noqa: E402

from app.contracts import HybridRetrieveIn, HybridRetrieveOut  # noqa: E402
from app.bulkhead import BulkheadFull, Pools  # noqa: E402
from mcp_servers.common import add_harness_tool  # noqa: E402
from app.harness import Caller, IdempotencyStore, Tier, ToolExecutionHarness, ToolFailure, ToolSpec  # noqa: E402
from mcp_servers.rag_pipeline import (  # noqa: E402
    DEFAULT_EMBED_DIR,
    DEFAULT_RERANK_DIR,
    HybridRetriever,
    IndexNotFound,
    IndexVersionError,
)


def register_rag_tools(harness: ToolExecutionHarness, retriever: HybridRetriever, pools: Pools) -> None:
    """Register hybrid_retrieve on any harness (the MCP server's own, or the API's in-process one: one implementation)."""

    async def _handler(m: HybridRetrieveIn, caller: Caller) -> HybridRetrieveOut:
        try:  # blocking ONNX/FAISS work runs off the event loop
            return await pools.rag.run_sync(  # own threads: a slow reranker can only exhaust the rag pool
                retriever.retrieve, m.tenant_id, m.query,
                top_k_sparse=m.top_k_sparse, top_k_dense=m.top_k_dense, rrf_k=m.rrf_k,
                rerank_k=m.rerank_k, score_threshold=m.score_threshold, domain=m.domain)
        except BulkheadFull:
            raise ToolFailure("E_RATE_LIMITED", "Retrieval is at capacity; retry with backoff.") from None
        except IndexNotFound:  # design 2.5: reject as not-found, never hint at other tenants
            raise ToolFailure("E_TENANT_MISMATCH", "No index available for this tenant.") from None
        except IndexVersionError:
            raise ToolFailure("E_UPSTREAM_UNAVAILABLE", "Index must be rebuilt for the current embedding model.") from None

    harness.register(ToolSpec("hybrid_retrieve", Tier.TIER_1_READ, HybridRetrieveIn, _handler))


def build_server(retriever: HybridRetriever, store: IdempotencyStore | None = None,
                 pools: Pools | None = None) -> MCPServer:
    server = MCPServer("hybrid-rag")
    harness = ToolExecutionHarness(store)
    register_rag_tools(harness, retriever, pools or Pools.from_env())

    add_harness_tool(
        server, harness, "hybrid_retrieve", HybridRetrieveIn, Tier.TIER_1_READ,
        "3-stage hybrid retrieval (BM25 + dense -> RRF -> cross-encoder). Abstains below the score threshold.")

    @server.resource("rag://chunks/{tenant_id}/{chunk_id}", mime_type="application/json")
    def chunk_resource(tenant_id: str, chunk_id: str) -> str:
        try:
            ch = retriever.get_chunk(tenant_id, chunk_id)
        except (IndexNotFound, IndexVersionError):
            raise ResourceError("chunk not found") from None  # identical for 'wrong tenant' and 'no such chunk'
        return json.dumps({"chunk_id": ch.chunk_id, "title": ch.title, "domain": ch.domain,
                           "source_uri": ch.source_uri, "text": ch.text})

    return server


def main() -> None:
    retriever = HybridRetriever(
        os.environ.get("RAG_INDEX_ROOT", "data/index"),
        os.environ.get("RAG_EMBED_DIR", DEFAULT_EMBED_DIR),
        os.environ.get("RAG_RERANK_DIR", DEFAULT_RERANK_DIR))
    build_server(retriever).run("stdio")


if __name__ == "__main__":
    main()

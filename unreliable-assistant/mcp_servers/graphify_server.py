"""mcp-graphify-vault: MCP server (stdio) returning the minimal subgraph of policy notes / code / schemas for a query.

Tool  graphify_subgraph  (TIER_1_READ, read-only)  - schema generated from app.contracts.GraphifySubgraphIn
One graph.json per tenant under GRAPHIFY_ROOT (default data/graph/<tenant>/graph.json). Every call goes through the
harness; graph lookups run on the triage bulkhead (own threads, never the rag pool).

Run:  GRAPHIFY_ROOT=data/graph uv run python mcp_servers/graphify_server.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python mcp_servers/graphify_server.py`

from mcp.server.mcpserver import MCPServer  # noqa: E402

from app.bulkhead import BulkheadFull, Pools  # noqa: E402
from app.contracts import TENANT_RE, GraphEdgeOut, GraphifySubgraphIn, GraphifySubgraphOut, GraphNodeOut  # noqa: E402
from app.harness import IdempotencyStore, Tier, ToolExecutionHarness, ToolFailure, ToolSpec  # noqa: E402
from mcp_servers.common import add_harness_tool  # noqa: E402
from mcp_servers.graphify import Graph  # noqa: E402

_TENANT = re.compile(TENANT_RE)


class GraphStore:
    """Lazily loads one Graph per tenant. The tenant id is regex-validated before touching the filesystem."""

    def __init__(self, root: str | Path, embedder: Any | None = None) -> None:
        self.root, self.embedder, self._graphs = Path(root), embedder, {}

    def for_tenant(self, tenant_id: str) -> Graph:
        if not _TENANT.match(tenant_id):
            raise FileNotFoundError("invalid tenant")
        if tenant_id not in self._graphs:
            path = self.root / tenant_id / "graph.json"
            if not path.exists():
                raise FileNotFoundError("no graph for tenant")
            self._graphs[tenant_id] = Graph.load(path, self.embedder)
        return self._graphs[tenant_id]


def register_graphify_tools(harness: ToolExecutionHarness, graphs: GraphStore, pools: Pools) -> None:
    async def handler(m: GraphifySubgraphIn, caller: Any) -> GraphifySubgraphOut:
        try:
            g = graphs.for_tenant(m.tenant_id)
        except FileNotFoundError:
            raise ToolFailure("E_TENANT_MISMATCH", "No graph available for this tenant.") from None
        try:
            sg = await pools.triage.run_sync(g.subgraph, m.query, m.depth, m.token_budget, m.hub_ids)
        except BulkheadFull:
            raise ToolFailure("E_RATE_LIMITED", "Graph service is at capacity; retry with backoff.") from None
        return GraphifySubgraphOut(
            anchor=sg.anchor, anchor_score=sg.anchor_score, anchors=sg.anchors, rendered=sg.rendered, tokens=sg.tokens,
            nodes=[GraphNodeOut(**{k: getattr(n, k) for k in GraphNodeOut.model_fields}) for n in sg.nodes],
            edges=[GraphEdgeOut(**e) for e in sg.edges])

    harness.register(ToolSpec("graphify_subgraph", Tier.TIER_1_READ, GraphifySubgraphIn, handler))


def build_server(graphs: GraphStore, store: IdempotencyStore | None = None, pools: Pools | None = None) -> MCPServer:
    server = MCPServer("graphify-vault")
    harness = ToolExecutionHarness(store)
    register_graphify_tools(harness, graphs, pools or Pools.from_env())
    add_harness_tool(server, harness, "graphify_subgraph", GraphifySubgraphIn, Tier.TIER_1_READ,
                     "Minimal connected subgraph (<= 2 hops) of policy notes, modules and schemas relevant to a query.")
    return server


def main() -> None:
    from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder

    embedder = OnnxEmbedder(os.environ.get("RAG_EMBED_DIR", DEFAULT_EMBED_DIR), threads=2) if os.environ.get("GRAPHIFY_DENSE", "1") == "1" else None
    build_server(GraphStore(os.environ.get("GRAPHIFY_ROOT", "data/graph"), embedder)).run("stdio")


if __name__ == "__main__":
    main()

"""Helpers shared by the MCP servers: contract-derived tool schemas, harness-routed tool registration."""
from __future__ import annotations

import inspect
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from app.harness import Caller, Tier, ToolExecutionHarness


def flat_signature(model: type[BaseModel]) -> inspect.Signature:
    """Keyword-only signature built from a Pydantic model so tools/list advertises the real contract."""
    params = [
        inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[f.annotation, f],
            default=inspect.Parameter.empty if f.is_required() else f.default)
        for name, f in model.model_fields.items()
    ]
    return inspect.Signature(params, return_annotation=dict[str, Any])


def add_harness_tool(server: MCPServer, harness: ToolExecutionHarness, name: str,
                     model: type[BaseModel], max_tier: Tier, description: str) -> None:
    """Expose an already-registered harness tool over MCP. Every call is routed through the harness."""

    async def tool(**kwargs: Any) -> dict[str, Any]:
        caller = Caller(tenant_id=kwargs["tenant_id"], user_id=kwargs["user_id"], max_tier=max_tier)
        res = await harness.call(name, kwargs, caller)
        if not res.ok:
            raise ToolError(res.error.model_dump_json())  # anticipated failure: message reaches the client
        return res.data  # type: ignore[return-value]

    sig = flat_signature(model)
    tool.__signature__ = sig  # type: ignore[attr-defined]
    tool.__annotations__ = {p.name: p.annotation for p in sig.parameters.values()}
    tool.__annotations__["return"] = dict[str, Any]
    server.add_tool(tool, name=name, description=description)

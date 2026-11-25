"""mcp-state-store: MCP server (streamable HTTP) for session facts and Tier 3 approval requests.

Tools: get_session_facts (TIER_1_READ), write_session_fact (TIER_2_COMPUTE), submit_transaction (TIER_3_MUTATE).
Every call is routed through ToolExecutionHarness. submit_transaction can only create a PENDING_APPROVAL
record; nothing here executes a transaction (design invariant B1).

Authorisation: this server has no per-caller identity yet (that arrives with Prompt 8, signed manifests / auth).
Until then the highest tier a connection may use is fixed per server instance by STATE_SERVER_MAX_TIER
(default 2). Tier 3 is therefore DENIED unless an operator deliberately starts an instance with 3,
and the server binds to 127.0.0.1 by default.

Run:  uv run python mcp_servers/state_server.py   (env: STATE_SERVER_HOST/PORT/MAX_TIER, see app/config.py)
"""
from __future__ import annotations

import asyncio
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python mcp_servers/state_server.py`

import psycopg  # noqa: E402
import psycopg_pool  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

from app.bulkhead import BulkheadFull, Pools  # noqa: E402
from app.config import pg_conninfo, redis_url  # noqa: E402
from app.contracts import GetSessionFactsIn, SubmitTransactionIn, WriteSessionFactIn  # noqa: E402
from app.harness import (  # noqa: E402
    IdempotencyStore,
    RedisIdempotencyStore,
    Tier,
    ToolExecutionHarness,
    ToolFailure,
    ToolSpec,
)
from app.pii import PiiDetected  # noqa: E402
from app.state_store import StateStore  # noqa: E402
from mcp_servers.common import add_harness_tool  # noqa: E402

ChunkValidator = Callable[[str, list[str]], Awaitable[bool]]


def _db_guard(fn):
    """Database outages become a structured, retryable error rather than a generic failure."""
    @wraps(fn)
    async def wrapper(m, caller):
        try:
            return await fn(m, caller)
        except PiiDetected as e:  # types and counts only: never the matched text
            raise ToolFailure("E_PII_DETECTED", "Personal data detected (" + ", ".join(sorted(e.counts)) + "); redact it first.") from None
        except BulkheadFull:
            raise ToolFailure("E_RATE_LIMITED", "State store is at capacity; retry with backoff.") from None
        except (psycopg.OperationalError, psycopg_pool.PoolTimeout):
            raise ToolFailure("E_UPSTREAM_UNAVAILABLE", "State store is unavailable.") from None
    return wrapper


def register_state_tools(harness: ToolExecutionHarness, store: StateStore, pools: Pools,
                         chunk_validator: ChunkValidator | None = None) -> None:
    """Register the three state tools on any harness (the MCP server's own, or the API's in-process one)."""

    @_db_guard
    async def get_facts(m: GetSessionFactsIn, caller):
        async with pools.db.slot():
            return await store.get_session_facts(m)

    @_db_guard
    async def write_fact(m: WriteSessionFactIn, caller):
        if chunk_validator is not None and not await chunk_validator(m.tenant_id, list(m.source_chunk_ids)):
            raise ToolFailure("E_NOT_GROUNDED", "source_chunk_ids do not reference retrievable chunks.")
        async with pools.db.slot():
            return await store.write_session_fact(m)

    @_db_guard
    async def submit(m: SubmitTransactionIn, caller):
        async with pools.db.slot():
            return await store.submit_transaction(m)

    harness.register(ToolSpec("get_session_facts", Tier.TIER_1_READ, GetSessionFactsIn, get_facts))
    harness.register(ToolSpec("write_session_fact", Tier.TIER_2_COMPUTE, WriteSessionFactIn, write_fact))
    harness.register(ToolSpec("submit_transaction", Tier.TIER_3_MUTATE, SubmitTransactionIn, submit))


def build_server(store: StateStore, idempotency: IdempotencyStore | None = None,
                 max_tier: Tier = Tier.TIER_2_COMPUTE, chunk_validator: ChunkValidator | None = None,
                 pools: Pools | None = None) -> MCPServer:
    server = MCPServer("state-store")
    harness = ToolExecutionHarness(idempotency)
    register_state_tools(harness, store, pools or Pools.from_env(), chunk_validator)

    add_harness_tool(server, harness, "get_session_facts", GetSessionFactsIn, max_tier,
                     "Read verified, redacted facts stored for a session.")
    add_harness_tool(server, harness, "write_session_fact", WriteSessionFactIn, max_tier,
                     "Persist a verified fact, grounded in retrieved chunk ids.")
    add_harness_tool(server, harness, "submit_transaction", SubmitTransactionIn, max_tier,
                     "Request a state-changing action. Always returns PENDING_APPROVAL; a human must approve.")
    return server


async def amain() -> None:
    import redis.asyncio as aioredis

    store = await StateStore.create(pg_conninfo(), max_size=int(os.environ.get("DB_POOL", "20")))
    server = build_server(store, RedisIdempotencyStore(aioredis.from_url(redis_url())),
                          Tier(int(os.environ.get("STATE_SERVER_MAX_TIER", "2"))))
    try:
        await server.run_streamable_http_async(
            host=os.environ.get("STATE_SERVER_HOST", "127.0.0.1"),
            port=int(os.environ.get("STATE_SERVER_PORT", "8081")))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(amain())

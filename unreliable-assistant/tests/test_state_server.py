import json
import uuid

import pytest
from mcp.client import Client
from psycopg_pool import AsyncConnectionPool

from app.harness import Tier
from app.state_store import StateStore
from mcp_servers.state_server import build_server
from tests.test_state_store import store  # noqa: F401  (async fixture: pooled StateStore on the throwaway DB)


def tenants():
    t = uuid.uuid4().hex[:8]
    return f"ta-{t}", f"tb-{t}"


def base(tenant, **kw):
    return {"tenant_id": tenant, "user_id": "u1", "request_id": str(uuid.uuid4()), **kw}


def fact_args(tenant, session, **kw):
    return base(tenant, session_id=session, fact_key="leave.days", value="25",
                source_chunk_ids=["pto-0000-ab12cd34"], verified=True, **kw)


def txn_args(tenant, session, **kw):
    return base(tenant, session_id=session, action="TRANSFER", account_id="ACC-12345678", amount="1250.50",
                currency="USD", justification="Quarterly settlement for vendor invoice 4471", **kw)


def err(result):
    assert result.is_error
    text = " ".join(getattr(b, "text", "") for b in result.content)
    return json.loads(text[text.index("{"):])  # the SDK prefixes "Error executing tool <name>: "


async def test_tools_list_matches_contracts(store):  # noqa: F811
    async with Client(build_server(store)) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == {"get_session_facts", "write_session_fact", "submit_transaction"}
    assert tools["submit_transaction"].input_schema["properties"]["account_id"]["pattern"] == r"^ACC-\d{8}$"
    assert tools["write_session_fact"].input_schema["properties"]["verified"]["const"] is True


async def test_write_then_read_facts_and_isolation(store):  # noqa: F811
    a, b = tenants()
    s = str(uuid.uuid4())
    async with Client(build_server(store)) as c:
        w = await c.call_tool("write_session_fact", fact_args(a, s))
        assert not w.is_error and len(w.structured_content["audit_hash"]) == 64
        r = await c.call_tool("get_session_facts", base(a, session_id=s))
        assert [f["value"] for f in r.structured_content["facts"]] == ["25"]
        other = await c.call_tool("get_session_facts", base(b, session_id=s))  # same session id, other tenant
        assert other.structured_content["facts"] == []


async def test_tier3_is_denied_by_default_and_creates_nothing(store):  # noqa: F811
    a, _ = tenants()
    s = str(uuid.uuid4())
    async with Client(build_server(store)) as c:  # default max_tier = 2
        res = await c.call_tool("submit_transaction", txn_args(a, s))
    e = err(res)
    assert e["code"] == "E_PERMISSION_DENIED" and e["rpc_code"] == -32003
    assert (await store.verify_chain(a)) == (True, None, 0)  # nothing was written, not even an audit row


async def test_tier3_when_enabled_only_ever_returns_pending_approval(store):  # noqa: F811
    a, _ = tenants()
    s = str(uuid.uuid4())
    args = txn_args(a, s)
    async with Client(build_server(store, max_tier=Tier.TIER_3_MUTATE)) as c:
        first = await c.call_tool("submit_transaction", args)
        replay = await c.call_tool("submit_transaction", args)  # same request_id
    assert first.structured_content["status"] == "PENDING_APPROVAL"
    assert replay.structured_content == first.structured_content  # idempotent: one approval
    row = await store.get_approval(a, first.structured_content["approval_id"])
    assert row["status"] == "PENDING_APPROVAL" and row["approver"] is None


async def test_ungrounded_fact_is_rejected(store):  # noqa: F811
    a, _ = tenants()

    async def nothing_is_grounded(tenant, chunk_ids):
        return False

    async with Client(build_server(store, chunk_validator=nothing_is_grounded)) as c:
        res = await c.call_tool("write_session_fact", fact_args(a, str(uuid.uuid4())))
    assert err(res)["code"] == "E_NOT_GROUNDED"
    assert (await store.verify_chain(a)) == (True, None, 0)


async def test_unverified_or_malformed_facts_never_reach_the_database(store):  # noqa: F811
    a, _ = tenants()
    s = str(uuid.uuid4())
    async with Client(build_server(store)) as c:
        assert (await c.call_tool("write_session_fact", fact_args(a, s) | {"verified": False})).is_error
        assert (await c.call_tool("write_session_fact", fact_args(a, s) | {"source_chunk_ids": []})).is_error
        assert (await c.call_tool("write_session_fact", fact_args(a, s) | {"fact_key": "BAD KEY"})).is_error
        assert (await c.call_tool("submit_transaction", txn_args(a, s) | {"account_id": "12345678"})).is_error
    assert (await store.verify_chain(a)) == (True, None, 0)


async def test_database_outage_is_a_structured_retryable_error():
    bad = AsyncConnectionPool("host=127.0.0.1 port=1 dbname=x user=x password=x connect_timeout=1",
                              min_size=0, max_size=1, timeout=1, open=False)
    await bad.open(wait=False)
    try:
        a, _ = tenants()
        async with Client(build_server(StateStore(bad))) as c:
            res = await c.call_tool("get_session_facts", base(a, session_id=str(uuid.uuid4())))
        e = err(res)
        assert e["code"] == "E_UPSTREAM_UNAVAILABLE" and e["retryable"] is True
    finally:
        await bad.close()

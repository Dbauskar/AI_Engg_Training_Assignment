import json
import uuid

import pytest
from mcp.client import Client

from mcp_servers.rag_server import build_server
from tests.test_rag_pipeline import TENANT, needs_models, retriever  # noqa: F401  (session fixture)

# NOTE: each test opens its own Client with `async with`. An async-generator fixture would enter and exit the
# anyio cancel scope in different tasks and error on teardown.


def args(**kw):
    return {"tenant_id": TENANT, "user_id": "u1", "request_id": str(uuid.uuid4()),
            "query": "How long is the battery warranty for SKU-205?", **kw}


def payload(result):
    assert not result.is_error, result.content
    return result.structured_content


def error_text(result):
    assert result.is_error
    return " ".join(getattr(b, "text", "") for b in result.content)


@needs_models
async def test_tools_list_advertises_contract_schema(retriever):  # noqa: F811
    async with Client(build_server(retriever)) as client:
        tools = (await client.list_tools()).tools
    t = next(t for t in tools if t.name == "hybrid_retrieve")
    props = t.input_schema["properties"]
    assert props["tenant_id"]["pattern"] == "^[a-z0-9][a-z0-9_-]{2,39}$"
    assert props["rerank_k"]["maximum"] == 10 and props["score_threshold"]["default"] == 0.42
    assert set(t.input_schema["required"]) == {"tenant_id", "user_id", "request_id", "query"}


@needs_models
async def test_call_returns_grounded_chunks(retriever):  # noqa: F811
    async with Client(build_server(retriever)) as client:
        out = payload(await client.call_tool("hybrid_retrieve", args()))
    assert out["abstain"] is False
    assert out["chunks"][0]["chunk_id"].startswith("warranty-")
    assert set(out["stage_latency_ms"]) == {"bm25", "dense", "rrf", "rerank"}


@needs_models
async def test_unanswerable_returns_abstain_not_error(retriever):  # noqa: F811
    async with Client(build_server(retriever)) as client:
        out = payload(await client.call_tool("hybrid_retrieve", args(query="How do I bake sourdough bread?")))
    assert out["abstain"] is True and out["chunks"] == []


@needs_models
async def test_replayed_request_id_returns_identical_stored_result(retriever):  # noqa: F811
    a = args()
    async with Client(build_server(retriever)) as client:
        first = payload(await client.call_tool("hybrid_retrieve", a))
        second = payload(await client.call_tool("hybrid_retrieve", a))
    assert first == second  # including stage latencies: served from the idempotency store, not recomputed


@needs_models
async def test_malformed_arguments_rejected_before_retrieval(retriever):  # noqa: F811
    async with Client(build_server(retriever)) as client:
        assert (await client.call_tool("hybrid_retrieve", args(tenant_id="BAD TENANT!"))).is_error
        assert (await client.call_tool("hybrid_retrieve", args(rerank_k=99))).is_error
        assert (await client.call_tool("hybrid_retrieve", args(query=""))).is_error
        assert (await client.call_tool("hybrid_retrieve", args(request_id="not-a-uuid"))).is_error


@needs_models
async def test_unknown_tenant_gets_structured_not_found_error(retriever):  # noqa: F811
    async with Client(build_server(retriever)) as client:
        res = await client.call_tool("hybrid_retrieve", args(tenant_id="no-such-tenant"))
    body = error_text(res)
    assert "E_TENANT_MISMATCH" in body and "retryable\":false" in body.replace(" ", "")
    assert "E_INTERNAL" not in body  # an expected condition, not a server fault


@needs_models
async def test_chunk_resource_is_tenant_scoped(retriever):  # noqa: F811
    cid = retriever.index_for(TENANT).chunks[0].chunk_id
    async with Client(build_server(retriever)) as client:
        ok = await client.read_resource(f"rag://chunks/{TENANT}/{cid}")
        body = json.loads(ok.contents[0].text)
        assert body["chunk_id"] == cid and body["text"]
        with pytest.raises(Exception):  # same chunk id, other tenant -> not found
            await client.read_resource(f"rag://chunks/other-co/{cid}")
        with pytest.raises(Exception):  # malformed chunk id
            await client.read_resource(f"rag://chunks/{TENANT}/not-a-chunk-id")
        templates = (await client.list_resource_templates()).resource_templates
    assert any(t.uri_template == "rag://chunks/{tenant_id}/{chunk_id}" for t in templates)

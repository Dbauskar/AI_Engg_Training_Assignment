import asyncio
import uuid
from typing import Annotated

import pytest
from pydantic import BaseModel, StringConstraints

from app.contracts import SubmitTransactionIn, SubmitTransactionOut, ToolInputBase
from app.harness import (
    Caller,
    InMemoryIdempotencyStore,
    RedisIdempotencyStore,
    Tier,
    ToolExecutionHarness,
    ToolSpec,
    idempotency_key,
)

TENANT = "acme-corp"


class InventoryIn(ToolInputBase):
    sku: Annotated[str, StringConstraints(pattern=r"^SKU-\d{3}$")]
    lot: Annotated[str, StringConstraints(pattern=r"^LOT-\d{5}$")] | None = None


class InventoryOut(BaseModel):
    sku: str
    lot: str | None
    on_hand: int


def base(**kw):
    return {"tenant_id": TENANT, "user_id": "u1", "request_id": str(uuid.uuid4()), **kw}


def caller(tier=Tier.TIER_3_MUTATE, tenant=TENANT):
    return Caller(tenant_id=tenant, user_id="u1", max_tier=tier)


@pytest.fixture
def h():
    calls = {"inventory": 0, "txn": 0}
    harness = ToolExecutionHarness(InMemoryIdempotencyStore())

    async def inventory(m: InventoryIn, c: Caller):
        calls["inventory"] += 1
        return InventoryOut(sku=m.sku, lot=m.lot, on_hand=7)

    async def txn(m: SubmitTransactionIn, c: Caller):
        calls["txn"] += 1
        return SubmitTransactionOut(
            status="PENDING_APPROVAL", approval_id=str(uuid.uuid4()), audit_hash="ab" * 32
        )

    harness.register(ToolSpec("get_inventory", Tier.TIER_1_READ, InventoryIn, inventory))
    harness.register(ToolSpec("submit_transaction", Tier.TIER_3_MUTATE, SubmitTransactionIn, txn))
    harness.calls = calls
    return harness


# --- 1. valid call -----------------------------------------------------------
async def test_valid_call(h):
    r = await h.call("get_inventory", base(sku="SKU-099"), caller(Tier.TIER_1_READ))
    assert r.ok and not r.repaired and not r.replayed
    assert r.data == {"sku": "SKU-099", "lot": None, "on_hand": 7}
    assert h.stats.first_pass_rate == 1.0


# --- 2. repairable call ------------------------------------------------------
@pytest.mark.parametrize("bad,good", [("SKU99", "SKU-099"), ("sku 7", "SKU-007"), (" SKU-5 ", "SKU-005")])
async def test_sku_auto_repair(h, bad, good):
    r = await h.call("get_inventory", base(sku=bad), caller(Tier.TIER_1_READ))
    assert r.ok and r.repaired
    assert r.data["sku"] == good


async def test_lot_auto_repair(h):
    r = await h.call("get_inventory", base(sku="SKU-001", lot="lot 44"), caller(Tier.TIER_1_READ))
    assert r.ok and r.repaired and r.data["lot"] == "LOT-00044"


async def test_transaction_amount_and_currency_repair(h):
    args = base(session_id=str(uuid.uuid4()), action="TRANSFER", account_id="ACC-12345678",
                amount=1250.5, currency=" usd ", justification="Quarterly settlement")
    r = await h.call("submit_transaction", args, caller())
    assert r.ok and r.repaired
    assert r.data["status"] == "PENDING_APPROVAL"


async def test_repair_stats(h):
    await h.call("get_inventory", base(sku="SKU-001"), caller(Tier.TIER_1_READ))
    await h.call("get_inventory", base(sku="SKU2"), caller(Tier.TIER_1_READ))
    await h.call("get_inventory", base(sku="garbage"), caller(Tier.TIER_1_READ))
    s = h.stats
    assert (s.calls, s.first_pass_valid, s.repair_attempted, s.repair_succeeded) == (3, 1, 2, 1)
    assert s.repair_success_rate == 0.5


# --- 3. unrepairable call ----------------------------------------------------
async def test_unrepairable_returns_structured_error_without_echoing_input(h):
    secret = "alice@example.com"
    r = await h.call("get_inventory", base(sku=secret), caller(Tier.TIER_1_READ))
    assert not r.ok and r.error.code == "E_SCHEMA_INVALID" and r.error.rpc_code == -32602
    assert r.error.fields == ["sku"]
    assert secret not in r.model_dump_json()
    assert h.calls["inventory"] == 0


async def test_money_is_never_rounded(h):
    args = base(session_id=str(uuid.uuid4()), action="TRANSFER", account_id="ACC-12345678",
                amount="10.999", currency="USD", justification="Quarterly settlement")
    r = await h.call("submit_transaction", args, caller())
    assert not r.ok and r.error.code == "E_SCHEMA_INVALID" and "amount" in r.error.fields


async def test_extra_fields_and_unknown_tool_rejected(h):
    r = await h.call("get_inventory", base(sku="SKU-001", admin=True), caller(Tier.TIER_1_READ))
    assert not r.ok and r.error.code == "E_SCHEMA_INVALID"
    r = await h.call("does_not_exist", base(), caller())
    assert not r.ok and r.error.code == "E_SCHEMA_INVALID"


# --- 4. wrong tier -----------------------------------------------------------
async def test_wrong_tier_denied_and_handler_not_run(h):
    args = base(session_id=str(uuid.uuid4()), action="ORDER", account_id="ACC-12345678",
                justification="Restock order for Q4")
    r = await h.call("submit_transaction", args, caller(Tier.TIER_2_COMPUTE))
    assert not r.ok and r.error.code == "E_PERMISSION_DENIED" and r.error.rpc_code == -32003
    assert h.calls["txn"] == 0


# --- 5. replayed idempotency key --------------------------------------------
async def test_replay_returns_stored_result_without_rerunning_handler(h):
    args = base(session_id=str(uuid.uuid4()), action="ORDER", account_id="ACC-12345678",
                justification="Restock order for Q4")
    first = await h.call("submit_transaction", args, caller())
    second = await h.call("submit_transaction", args, caller())
    assert first.ok and second.ok and second.replayed and not first.replayed
    assert second.data == first.data  # same approval_id: no duplicate transaction
    assert h.calls["txn"] == 1


async def test_concurrent_duplicates_run_handler_once(h):
    args = base(session_id=str(uuid.uuid4()), action="ORDER", account_id="ACC-12345678",
                justification="Restock order for Q4")
    results = await asyncio.gather(*[h.call("submit_transaction", args, caller()) for _ in range(20)])
    assert h.calls["txn"] == 1
    assert all(r.ok or r.error.code == "E_IDEMPOTENT_REPLAY" for r in results)


async def test_handler_failure_releases_key_so_retry_can_succeed(h):
    boom = {"n": 0}

    async def flaky(m: InventoryIn, c: Caller):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("db exploded with alice@example.com")
        return InventoryOut(sku=m.sku, lot=None, on_hand=1)

    h2 = ToolExecutionHarness()
    h2.register(ToolSpec("flaky", Tier.TIER_1_READ, InventoryIn, flaky))
    args = base(sku="SKU-001")
    r1 = await h2.call("flaky", args, caller(Tier.TIER_1_READ))
    assert not r1.ok and r1.error.code == "E_INTERNAL" and "alice" not in r1.model_dump_json()
    r2 = await h2.call("flaky", args, caller(Tier.TIER_1_READ))
    assert r2.ok


# --- tenant guard (CLAUDE.md rule 3) ----------------------------------------
async def test_missing_tenant_rejected(h):
    args = base(sku="SKU-001")
    del args["tenant_id"]
    r = await h.call("get_inventory", args, caller(Tier.TIER_1_READ))
    assert not r.ok and r.error.code == "E_TENANT_MISSING"
    assert h.calls["inventory"] == 0


async def test_tenant_mismatch_rejected(h):
    r = await h.call("get_inventory", base(sku="SKU-001"), caller(Tier.TIER_1_READ, tenant="other-co"))
    assert not r.ok and r.error.code == "E_TENANT_MISMATCH"


async def test_same_request_id_in_two_tenants_does_not_collide():
    h2 = ToolExecutionHarness()

    async def ok(m, c):
        return {"tenant": m.tenant_id}

    h2.register(ToolSpec("t", Tier.TIER_1_READ, InventoryIn, ok))
    rid = str(uuid.uuid4())
    for tenant in ("acme-corp", "other-co"):
        a = {"tenant_id": tenant, "user_id": "u1", "request_id": rid, "sku": "SKU-001"}
        r = await h2.call("t", a, caller(Tier.TIER_1_READ, tenant=tenant))
        assert r.ok and not r.replayed and r.data == {"tenant": tenant}


# --- OTel span (CLAUDE.md rule 5) -------------------------------------------
async def test_emits_otel_span_with_outcome_attributes(h, spans):
    await h.call("get_inventory", base(sku="SKU99"), caller(Tier.TIER_1_READ))
    s = [x for x in spans.get_finished_spans() if x.name == "tool.harness"]
    assert len(s) == 1
    a = s[0].attributes
    assert a["tool.name"] == "get_inventory" and a["tenant.id"] == TENANT
    assert a["harness.ok"] is True and a["harness.repaired"] is True


async def test_span_never_contains_raw_input(h, spans):
    await h.call("get_inventory", base(sku="alice@example.com"), caller(Tier.TIER_1_READ))
    for sp in spans.get_finished_spans():
        assert "alice@example.com" not in repr(dict(sp.attributes)) + sp.name


# --- Redis-backed store (integration; skipped when Redis is unreachable) ----
@pytest.fixture
async def redis_store():
    redis = pytest.importorskip("redis.asyncio")
    client = redis.Redis(host="127.0.0.1", port=6379)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not reachable on 127.0.0.1:6379")
    yield RedisIdempotencyStore(client, ttl_s=30)
    await client.aclose()


async def test_redis_store_replay(redis_store):
    h2 = ToolExecutionHarness(redis_store)
    n = {"c": 0}

    async def ok(m, c):
        n["c"] += 1
        return {"v": n["c"]}

    h2.register(ToolSpec("t", Tier.TIER_1_READ, InventoryIn, ok))
    args = base(sku="SKU-001")
    a = await h2.call("t", args, caller(Tier.TIER_1_READ))
    b = await h2.call("t", args, caller(Tier.TIER_1_READ))
    assert a.ok and b.ok and b.replayed and n["c"] == 1
    await redis_store.release(idempotency_key(TENANT, args["request_id"]))


# --- cancellation (found while building the ReAct loop's per-step timeout) ---
async def test_cancelled_call_releases_its_idempotency_key(h):
    started = asyncio.Event()

    async def slow(m, c):
        started.set()
        await asyncio.sleep(5)

    h2 = ToolExecutionHarness()
    h2.register(ToolSpec("slow", Tier.TIER_1_READ, InventoryIn, slow))
    args = base(sku="SKU-001")
    task = asyncio.create_task(h2.call("slow", args, caller(Tier.TIER_1_READ)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # the same request_id must be usable again: it must not be stuck as "already in flight"
    assert await h2._store.claim(idempotency_key(TENANT, args["request_id"])) is True


def test_amount_repair_normalises_trailing_zeros_but_never_rounds():
    from app.harness import repair_amount
    assert repair_amount("1250.5000") == "1250.50"
    assert repair_amount("1,250.50") == "1250.50" and repair_amount(" 300 ") == "300"
    assert repair_amount("100.505") == "100.505"  # would need rounding: left alone, validation rejects it

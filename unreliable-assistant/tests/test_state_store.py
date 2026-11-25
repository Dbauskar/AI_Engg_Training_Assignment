import asyncio
import uuid

import psycopg
import pytest

from app.contracts import GetSessionFactsIn, SubmitTransactionIn, WriteSessionFactIn
from app.state_store import ALL_SQL, GENESIS, StateStore, digest


def ids():
    t = uuid.uuid4().hex[:8]
    return f"ta-{t}", f"tb-{t}", str(uuid.uuid4())  # tenant A, tenant B, session


def fact(tenant, session, key="leave.days", value="25", user="u1"):
    return WriteSessionFactIn(tenant_id=tenant, user_id=user, request_id=str(uuid.uuid4()), session_id=session,
                              fact_key=key, value=value, source_chunk_ids=["pto-0000-ab12cd34"], verified=True)


def txn(tenant, session, request_id=None, user="u1", **kw):
    d = dict(tenant_id=tenant, user_id=user, request_id=request_id or str(uuid.uuid4()), session_id=session,
             action="TRANSFER", account_id="ACC-12345678", amount="1250.50", currency="USD",
             justification="Quarterly settlement for vendor invoice 4471")
    d.update(kw)
    return SubmitTransactionIn(**d)


@pytest.fixture
async def store(pg_test_db):
    s = await StateStore.create(pg_test_db["app"], min_size=2, max_size=30)
    yield s
    await s.close()


def admin(pg_test_db):
    return psycopg.connect(pg_test_db["admin"], autocommit=True)


# ---------------- rule 3: tenant_id everywhere ----------------
def test_every_sql_statement_filters_on_tenant_id():
    assert len(ALL_SQL) >= 10
    missing = [name for name, q in ALL_SQL.items() if "tenant_id" not in q]
    assert missing == [], f"SQL without tenant_id: {missing}"


async def test_tenant_tx_rejects_missing_or_malformed_tenant(store):
    for bad in (None, "", "UPPER", "a b", "../x"):
        with pytest.raises(ValueError):
            async with store.tenant_tx(bad):
                pass


# ---------------- facts ----------------
async def test_fact_roundtrip_upsert_filter_and_limit(store):
    a, _, s = ids()
    await store.write_session_fact(fact(a, s, "leave.days", "25"))
    await store.write_session_fact(fact(a, s, "leave.carryover", "5"))
    await store.write_session_fact(fact(a, s, "leave.days", "26"))  # upsert: same key replaced
    got = await store.get_session_facts(GetSessionFactsIn(tenant_id=a, user_id="u1", request_id=str(uuid.uuid4()), session_id=s))
    assert {f.fact_key: f.value for f in got.facts} == {"leave.days": "26", "leave.carryover": "5"}
    assert all(f.verified for f in got.facts)
    only = await store.get_session_facts(GetSessionFactsIn(
        tenant_id=a, user_id="u1", request_id=str(uuid.uuid4()), session_id=s, keys=["leave.days"]))
    assert [f.fact_key for f in only.facts] == ["leave.days"]
    one = await store.get_session_facts(GetSessionFactsIn(
        tenant_id=a, user_id="u1", request_id=str(uuid.uuid4()), session_id=s, limit=1))
    assert len(one.facts) == 1


async def test_facts_are_invisible_to_other_tenants_even_with_same_session_id(store):
    a, b, s = ids()
    await store.write_session_fact(fact(a, s, value="A-only"))
    got = await store.get_session_facts(GetSessionFactsIn(tenant_id=b, user_id="u1", request_id=str(uuid.uuid4()), session_id=s))
    assert got.facts == []


# ---------------- row-level security (defence in depth) ----------------
async def test_rls_blocks_cross_tenant_rows_even_without_a_where_clause(store, pg_test_db):
    a, b, s = ids()
    await store.write_session_fact(fact(a, s, value="A"))
    await store.write_session_fact(fact(b, s, value="B"))
    with psycopg.connect(pg_test_db["app"]) as c:  # raw connection as the limited app role
        assert c.execute("SELECT count(*) FROM session_facts").fetchone()[0] == 0  # no tenant set: fail closed
        c.execute("SELECT set_config('app.tenant_id', %s, false)", (a,))
        rows = c.execute("SELECT tenant_id, value FROM session_facts").fetchall()  # NOTE: no WHERE
        assert rows == [(a, "A")]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):  # WITH CHECK: cannot write a row for tenant B
            c.execute("INSERT INTO session_facts VALUES (%s, %s, 'x.key', 'v', true, ARRAY['pto-0000-ab12cd34'], 'now')", (b, s))


async def test_app_role_cannot_tamper_with_audit_or_approve_requests(store, pg_test_db):
    a, _, s = ids()
    await store.write_session_fact(fact(a, s))
    out = await store.submit_transaction(txn(a, s))
    with psycopg.connect(pg_test_db["app"], autocommit=True) as c:
        c.execute("SELECT set_config('app.tenant_id', %s, false)", (a,))
        for stmt in ("UPDATE audit_log SET payload = '{}'::jsonb", "DELETE FROM audit_log",
                     "UPDATE approvals SET status = 'APPROVED'", "DELETE FROM approvals"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                c.execute(stmt)
    assert (await store.get_approval(a, out.approval_id))["status"] == "PENDING_APPROVAL"


# ---------------- audit hash chain ----------------
async def test_audit_chain_links_and_verifies(store, pg_test_db):
    a, _, s = ids()
    hs = [(await store.write_session_fact(fact(a, s, f"k.fact{i}", str(i)))).audit_hash for i in range(3)]
    assert len(set(hs)) == 3
    ok, broken, n = await store.verify_chain(a)
    assert (ok, broken, n) == (True, None, 3)
    with admin(pg_test_db) as c:
        rows = c.execute("SELECT seq, prev_hash, hash FROM audit_log WHERE tenant_id=%s ORDER BY seq", (a,)).fetchall()
    assert rows[0][1] == GENESIS and rows[1][1] == rows[0][2] and rows[2][1] == rows[1][2]
    assert [r[2] for r in rows] == hs


async def test_concurrent_writes_keep_one_valid_gapless_chain(store, pg_test_db):
    a, _, s = ids()
    await asyncio.gather(*[store.write_session_fact(fact(a, s, f"k.fact{i}", str(i))) for i in range(40)])
    ok, broken, n = await store.verify_chain(a)
    assert ok and broken is None and n == 40
    with admin(pg_test_db) as c:
        seqs = [r[0] for r in c.execute("SELECT seq FROM audit_log WHERE tenant_id=%s ORDER BY seq", (a,))]
    assert seqs == list(range(1, 41))


async def test_chains_are_per_tenant(store):
    a, b, s = ids()
    await store.write_session_fact(fact(a, s))
    await store.write_session_fact(fact(b, s))
    assert await store.verify_chain(a) == (True, None, 1)
    assert await store.verify_chain(b) == (True, None, 1)


async def test_tampering_and_deletion_are_detected(store, pg_test_db):
    a, _, s = ids()
    for i in range(4):
        await store.write_session_fact(fact(a, s, f"k.fact{i}", str(i)))
    with admin(pg_test_db) as c:  # superuser (bypasses RLS): the attacker model
        c.execute("UPDATE audit_log SET payload = jsonb_set(payload, '{fact_key}', '\"k.forged\"') WHERE tenant_id=%s AND seq=2", (a,))
    assert await store.verify_chain(a) == (False, 2, 4)
    b_t = ids()[0]
    for i in range(3):
        await store.write_session_fact(fact(b_t, s, f"k.fact{i}", str(i)))
    with admin(pg_test_db) as c:
        c.execute("DELETE FROM audit_log WHERE tenant_id=%s AND seq=2", (b_t,))
    ok, broken, _ = await store.verify_chain(b_t)
    assert not ok and broken == 3  # gap in the sequence


async def test_audit_trail_never_contains_raw_values(store, pg_test_db):
    a, _, s = ids()
    secret, justification = "super-secret-value-xyz", "Settlement for the confidential Acme merger fee"
    await store.write_session_fact(fact(a, s, value=secret))
    await store.submit_transaction(txn(a, s, justification=justification, account_id="ACC-99887766"))
    with admin(pg_test_db) as c:
        blob = " ".join(str(r) for r in c.execute("SELECT payload FROM audit_log WHERE tenant_id=%s", (a,)))
    for raw in (secret, justification, "ACC-99887766"):
        assert raw not in blob
    assert digest(secret) in blob and digest("ACC-99887766") in blob


# ---------------- approvals ----------------
async def test_submit_transaction_only_creates_a_pending_record(store):
    a, _, s = ids()
    out = await store.submit_transaction(txn(a, s))
    assert out.status == "PENDING_APPROVAL" and len(out.audit_hash) == 64
    row = await store.get_approval(a, out.approval_id)
    assert row["status"] == "PENDING_APPROVAL" and row["approver"] is None and row["requester"] == "u1"
    assert row["action"] == "TRANSFER" and row["amount"] == "1250.50"


async def test_resubmitting_same_request_id_is_idempotent_in_the_database(store, pg_test_db):
    a, _, s = ids()
    rid = str(uuid.uuid4())
    first = await store.submit_transaction(txn(a, s, request_id=rid))
    again = await store.submit_transaction(txn(a, s, request_id=rid))  # e.g. the Redis idempotency cache was lost
    assert again.approval_id == first.approval_id and again.audit_hash == first.audit_hash
    with admin(pg_test_db) as c:
        assert c.execute("SELECT count(*) FROM approvals WHERE tenant_id=%s", (a,)).fetchone()[0] == 1
        assert c.execute("SELECT count(*) FROM audit_log WHERE tenant_id=%s AND event_type='txn.submit'", (a,)).fetchone()[0] == 1


async def test_parallel_duplicate_submissions_create_one_approval(store, pg_test_db):
    a, _, s = ids()
    rid = str(uuid.uuid4())
    outs = await asyncio.gather(*[store.submit_transaction(txn(a, s, request_id=rid)) for _ in range(15)])
    assert len({o.approval_id for o in outs}) == 1
    with admin(pg_test_db) as c:
        assert c.execute("SELECT count(*) FROM approvals WHERE tenant_id=%s", (a,)).fetchone()[0] == 1
    assert (await store.verify_chain(a))[0] is True


async def test_requester_cannot_be_their_own_approver(store, pg_test_db):
    a, _, s = ids()
    out = await store.submit_transaction(txn(a, s, user="alice"))
    with admin(pg_test_db) as c:
        with pytest.raises(psycopg.errors.CheckViolation):
            c.execute("UPDATE approvals SET approver='alice', status='APPROVED' WHERE tenant_id=%s AND approval_id=%s", (a, out.approval_id))
        c.execute("UPDATE approvals SET approver='bob', status='APPROVED' WHERE tenant_id=%s AND approval_id=%s", (a, out.approval_id))  # a different person is fine


async def test_approvals_are_tenant_scoped(store):
    a, b, s = ids()
    out = await store.submit_transaction(txn(a, s))
    assert await store.get_approval(b, out.approval_id) is None


# ---------------- OTel (rule 5) ----------------
async def test_store_operations_emit_spans(store, spans):
    a, _, s = ids()
    await store.write_session_fact(fact(a, s))
    await store.get_session_facts(GetSessionFactsIn(tenant_id=a, user_id="u1", request_id=str(uuid.uuid4()), session_id=s))
    await store.submit_transaction(txn(a, s))
    names = {sp.name for sp in spans.get_finished_spans()}
    assert {"state.write_session_fact", "state.get_session_facts", "state.submit_transaction"} <= names
    assert all(sp.attributes.get("tenant.id") == a for sp in spans.get_finished_spans() if sp.name.startswith("state."))

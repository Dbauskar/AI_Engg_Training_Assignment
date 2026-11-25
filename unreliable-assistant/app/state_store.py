"""Tenant-scoped async Postgres state store: session facts, hash-chained audit log, approvals.

Rules enforced here (CLAUDE.md): tenant_id is required for every query (rule 3) and every SQL statement
below references it (checked by tests/test_state_store.py::test_every_sql_statement_filters_on_tenant_id).
The audit trail never stores raw values: free text and account numbers are recorded as SHA-256 digests (rule 4).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from opentelemetry import trace
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.pii import PiiDetector, shared_detector
from app.contracts import (
    TENANT_RE,
    GetSessionFactsIn,
    GetSessionFactsOut,
    SessionFact,
    SubmitTransactionIn,
    SubmitTransactionOut,
    WriteSessionFactIn,
    WriteSessionFactOut,
)

_tracer = trace.get_tracer("unreliable_assistant.state_store")
_TENANT = re.compile(TENANT_RE)
GENESIS = "0" * 64

# ---- every statement is a named constant so it can be linted for tenant_id ---------------------------------
SQL_SET_TENANT = "SELECT set_config('app.tenant_id', %(tenant_id)s, true)"
SQL_AUDIT_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(%(tenant_id)s, 0))"
SQL_AUDIT_HEAD = ("SELECT seq, hash FROM audit_log WHERE tenant_id = %(tenant_id)s "
                  "ORDER BY seq DESC LIMIT 1")
SQL_AUDIT_INSERT = (
    "INSERT INTO audit_log (tenant_id, seq, event_type, actor, payload, ts, prev_hash, hash) "
    "VALUES (%(tenant_id)s, %(seq)s, %(event_type)s, %(actor)s, %(payload)s::jsonb, %(ts)s, %(prev_hash)s, %(hash)s)")
SQL_AUDIT_ALL = ("SELECT seq, event_type, actor, payload, ts, prev_hash, hash FROM audit_log "
                 "WHERE tenant_id = %(tenant_id)s ORDER BY seq")
SQL_AUDIT_FOR_APPROVAL = ("SELECT hash FROM audit_log WHERE tenant_id = %(tenant_id)s "
                          "AND event_type = 'txn.submit' AND payload->>'approval_id' = %(approval_id)s")
SQL_FACT_UPSERT = (
    "INSERT INTO session_facts (tenant_id, session_id, fact_key, value, verified, source_chunk_ids, created_at) "
    "VALUES (%(tenant_id)s, %(session_id)s, %(fact_key)s, %(value)s, true, %(chunks)s, %(ts)s) "
    "ON CONFLICT (tenant_id, session_id, fact_key) DO UPDATE SET value = EXCLUDED.value, "
    "source_chunk_ids = EXCLUDED.source_chunk_ids, created_at = EXCLUDED.created_at")
SQL_FACTS_GET = (
    "SELECT fact_key, value, verified, created_at FROM session_facts "
    "WHERE tenant_id = %(tenant_id)s AND session_id = %(session_id)s "
    "AND (%(keys)s::text[] IS NULL OR fact_key = ANY(%(keys)s::text[])) "
    "ORDER BY created_at, fact_key LIMIT %(limit)s")
SQL_APPROVAL_INSERT = (
    "INSERT INTO approvals (approval_id, tenant_id, request_id, session_id, requester, action, account_id, "
    "amount, currency, justification, status, created_at) "
    "VALUES (%(approval_id)s, %(tenant_id)s, %(request_id)s, %(session_id)s, %(requester)s, %(action)s, "
    "%(account_id)s, %(amount)s, %(currency)s, %(justification)s, 'PENDING_APPROVAL', %(ts)s) "
    "ON CONFLICT (tenant_id, request_id) DO NOTHING RETURNING approval_id")
SQL_APPROVAL_BY_REQUEST = ("SELECT approval_id FROM approvals WHERE tenant_id = %(tenant_id)s "
                           "AND request_id = %(request_id)s")
SQL_APPROVAL_GET = ("SELECT approval_id, status, requester, approver, action, amount, currency FROM approvals "
                    "WHERE tenant_id = %(tenant_id)s AND approval_id = %(approval_id)s")

ALL_SQL = {k: v for k, v in globals().items() if k.startswith("SQL_")}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def chain_hash(prev_hash: str, seq: int, event_type: str, actor: str, ts: str, payload: dict[str, Any]) -> str:
    return digest("|".join([prev_hash, str(seq), event_type, actor, ts, canonical(payload)]))


class StateStore:
    def __init__(self, pool: AsyncConnectionPool, pii: PiiDetector | None = None) -> None:
        self.pool = pool
        self.pii = pii or shared_detector()  # persistence guard: raw PII is refused, never stored (design Gate 13)

    @classmethod
    async def create(cls, conninfo: str, min_size: int = 2, max_size: int = 20, pii: PiiDetector | None = None) -> "StateStore":
        pool = AsyncConnectionPool(conninfo, min_size=min_size, max_size=max_size, open=False,
                                   kwargs={"row_factory": dict_row})
        await pool.open(wait=True, timeout=15)
        return cls(pool, pii)

    async def close(self) -> None:
        await self.pool.close()

    @asynccontextmanager
    async def tenant_tx(self, tenant_id: str) -> AsyncIterator[Any]:
        """Transaction bound to one tenant: sets the RLS variable first. Rejects a missing/invalid tenant."""
        if not isinstance(tenant_id, str) or not _TENANT.match(tenant_id):
            raise ValueError("tenant_id is required and must be well-formed")
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(SQL_SET_TENANT, {"tenant_id": tenant_id})
                yield conn

    # ---- audit log -----------------------------------------------------------------------------------------
    async def _append_audit(self, conn: Any, tenant_id: str, event_type: str, actor: str,
                            payload: dict[str, Any]) -> str:
        self.pii.assert_clean(payload, ner=False, ignore_ids=True)  # audit trails hold digests/counts/ids only
        await conn.execute(SQL_AUDIT_LOCK, {"tenant_id": tenant_id})  # serialise appends per tenant
        head = await (await conn.execute(SQL_AUDIT_HEAD, {"tenant_id": tenant_id})).fetchone()
        seq, prev = (head["seq"] + 1, head["hash"]) if head else (1, GENESIS)
        ts = _now()
        h = chain_hash(prev, seq, event_type, actor, ts, payload)
        await conn.execute(SQL_AUDIT_INSERT, {
            "tenant_id": tenant_id, "seq": seq, "event_type": event_type, "actor": actor,
            "payload": canonical(payload), "ts": ts, "prev_hash": prev, "hash": h})
        return h

    async def append_audit_event(self, tenant_id: str, actor: str, event_type: str, payload: dict[str, Any]) -> str:
        """Append one hash-chained audit event. Payloads must already be free of raw values (digests/counts only)."""
        async with self.tenant_tx(tenant_id) as conn:
            return await self._append_audit(conn, tenant_id, event_type, actor, payload)

    async def verify_chain(self, tenant_id: str) -> tuple[bool, int | None, int]:
        """Recompute the whole chain. Returns (ok, first_broken_seq, rows_checked)."""
        async with self.tenant_tx(tenant_id) as conn:
            rows = await (await conn.execute(SQL_AUDIT_ALL, {"tenant_id": tenant_id})).fetchall()
        prev, expected_seq = GENESIS, 1
        for r in rows:
            ok = (r["seq"] == expected_seq and r["prev_hash"] == prev and
                  r["hash"] == chain_hash(prev, r["seq"], r["event_type"], r["actor"], r["ts"], r["payload"]))
            if not ok:
                return False, r["seq"], len(rows)
            prev, expected_seq = r["hash"], expected_seq + 1
        return True, None, len(rows)

    # ---- tools ---------------------------------------------------------------------------------------------
    async def get_session_facts(self, m: GetSessionFactsIn) -> GetSessionFactsOut:
        with _tracer.start_as_current_span("state.get_session_facts") as sp:
            sp.set_attribute("tenant.id", m.tenant_id)
            async with self.tenant_tx(m.tenant_id) as conn:
                rows = await (await conn.execute(SQL_FACTS_GET, {
                    "tenant_id": m.tenant_id, "session_id": m.session_id,
                    "keys": m.keys, "limit": m.limit})).fetchall()
            sp.set_attribute("state.rows", len(rows))
            return GetSessionFactsOut(facts=[SessionFact(**r) for r in rows])

    async def write_session_fact(self, m: WriteSessionFactIn) -> WriteSessionFactOut:
        with _tracer.start_as_current_span("state.write_session_fact") as sp:
            sp.set_attribute("tenant.id", m.tenant_id)
            ts = _now()
            await asyncio.to_thread(self.pii.assert_clean, {"value": m.value})  # raises PiiDetected: nothing is written
            async with self.tenant_tx(m.tenant_id) as conn:
                await conn.execute(SQL_FACT_UPSERT, {
                    "tenant_id": m.tenant_id, "session_id": m.session_id, "fact_key": m.fact_key,
                    "value": m.value, "chunks": list(m.source_chunk_ids), "ts": ts})
                h = await self._append_audit(conn, m.tenant_id, "fact.write", m.user_id, {
                    "session_id": m.session_id, "fact_key": m.fact_key,
                    "value_sha256": digest(m.value), "source_chunk_ids": list(m.source_chunk_ids)})
            return WriteSessionFactOut(stored=True, fact_key=m.fact_key, audit_hash=h)

    async def submit_transaction(self, m: SubmitTransactionIn) -> SubmitTransactionOut:
        """Creates a PENDING_APPROVAL record. This method has no code path that executes anything."""
        with _tracer.start_as_current_span("state.submit_transaction") as sp:
            sp.set_attribute("tenant.id", m.tenant_id)
            await asyncio.to_thread(self.pii.assert_clean, {"justification": m.justification})
            async with self.tenant_tx(m.tenant_id) as conn:
                params = {"approval_id": str(uuid.uuid4()), "tenant_id": m.tenant_id, "request_id": m.request_id,
                          "session_id": m.session_id, "requester": m.user_id, "action": m.action,
                          "account_id": m.account_id, "amount": m.amount, "currency": m.currency,
                          "justification": m.justification, "ts": _now()}
                row = await (await conn.execute(SQL_APPROVAL_INSERT, params)).fetchone()
                if row is None:  # same request_id already stored (e.g. Redis idempotency cache was lost)
                    prior = await (await conn.execute(SQL_APPROVAL_BY_REQUEST, {
                        "tenant_id": m.tenant_id, "request_id": m.request_id})).fetchone()
                    aid = str(prior["approval_id"])
                    h = (await (await conn.execute(SQL_AUDIT_FOR_APPROVAL, {
                        "tenant_id": m.tenant_id, "approval_id": aid})).fetchone())["hash"]
                else:
                    aid = str(row["approval_id"])
                    h = await self._append_audit(conn, m.tenant_id, "txn.submit", m.user_id, {
                        "approval_id": aid, "session_id": m.session_id, "action": m.action,
                        "account_sha256": digest(m.account_id), "amount": m.amount, "currency": m.currency,
                        "justification_sha256": digest(m.justification)})
            sp.set_attribute("state.status", "PENDING_APPROVAL")
            return SubmitTransactionOut(status="PENDING_APPROVAL", approval_id=aid, audit_hash=h)

    async def get_approval(self, tenant_id: str, approval_id: str) -> dict[str, Any] | None:
        async with self.tenant_tx(tenant_id) as conn:
            return await (await conn.execute(SQL_APPROVAL_GET, {
                "tenant_id": tenant_id, "approval_id": approval_id})).fetchone()

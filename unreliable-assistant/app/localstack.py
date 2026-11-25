"""Service-free local stack: the REAL pipeline (hooks, triage, hybrid RAG, verification, harness, cache logic) with
in-memory stand-ins for Postgres and Redis. Used by the Streamlit workbench and the local load test when docker/Postgres/Redis
are not available.

What it is NOT: it proves nothing about Postgres row-level security, Redis latency/TTL/clustering, or network hops. Any number
measured on it must be labelled "local in-memory stack" (see scripts/kpi_gate.py scale labelling).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from unittest import mock

from app.contracts import (GetSessionFactsIn, GetSessionFactsOut, SessionFact, SubmitTransactionIn, SubmitTransactionOut,
                           WriteSessionFactIn, WriteSessionFactOut)
from app.runtime import Runtime, Settings, build_runtime
from evals.fake_redis import FakeRedis

GENESIS = "0" * 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _h(prev: str, seq: int, event: str, actor: str, ts: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prev}|{seq}|{event}|{actor}|{ts}|{body}".encode()).hexdigest()


class _Pool:
    """Just enough of psycopg's pool for the readiness probe."""

    def connection(self):
        class _C:
            async def __aenter__(s):
                return s

            async def __aexit__(s, *a):
                return False

            async def execute(s, *a, **k):
                return None
        return _C()


class InMemoryStateStore:
    """Same public methods as app.state_store.StateStore. Every structure is keyed by tenant_id first (CLAUDE.md rule 3)."""

    def __init__(self) -> None:
        self.pool = _Pool()
        self.facts: dict[tuple[str, str, str], SessionFact] = {}
        self.audit: dict[str, list[dict[str, Any]]] = {}
        self.approvals: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls, *a: Any, **k: Any) -> "InMemoryStateStore":
        return cls()

    async def close(self) -> None:
        return None

    @staticmethod
    def _need_tenant(tenant_id: str) -> None:
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id is required")

    def _append(self, tenant_id: str, event: str, actor: str, payload: dict[str, Any]) -> str:
        rows = self.audit.setdefault(tenant_id, [])
        prev = rows[-1]["hash"] if rows else GENESIS
        seq, ts = len(rows) + 1, _now()
        h = _h(prev, seq, event, actor, ts, payload)
        rows.append({"seq": seq, "event_type": event, "actor": actor, "ts": ts, "payload": payload, "prev_hash": prev, "hash": h})
        return h

    async def append_audit_event(self, tenant_id: str, actor: str, event_type: str, payload: dict[str, Any]) -> str:
        self._need_tenant(tenant_id)
        async with self._lock:
            return self._append(tenant_id, event_type, actor, payload)

    async def verify_chain(self, tenant_id: str) -> tuple[bool, int | None, int]:
        self._need_tenant(tenant_id)
        prev = GENESIS
        rows = self.audit.get(tenant_id, [])
        for i, r in enumerate(rows, 1):
            if r["seq"] != i or r["prev_hash"] != prev or r["hash"] != _h(prev, i, r["event_type"], r["actor"], r["ts"], r["payload"]):
                return False, r["seq"], len(rows)
            prev = r["hash"]
        return True, None, len(rows)

    async def get_session_facts(self, m: GetSessionFactsIn) -> GetSessionFactsOut:
        self._need_tenant(m.tenant_id)
        out = [f for (t, s, k), f in self.facts.items() if t == m.tenant_id and s == str(m.session_id) and (not m.keys or k in m.keys)]
        return GetSessionFactsOut(facts=out[: m.limit])

    async def write_session_fact(self, m: WriteSessionFactIn) -> WriteSessionFactOut:
        self._need_tenant(m.tenant_id)
        async with self._lock:
            self.facts[(m.tenant_id, str(m.session_id), m.fact_key)] = SessionFact(
                fact_key=m.fact_key, value=m.value, verified=m.verified, created_at=_now())
            h = self._append(m.tenant_id, "fact.write", m.user_id, {
                "session_id": str(m.session_id), "fact_key": m.fact_key,
                "value_sha256": hashlib.sha256(m.value.encode()).hexdigest(), "source_chunk_ids": list(m.source_chunk_ids)})
        return WriteSessionFactOut(stored=True, fact_key=m.fact_key, audit_hash=h)

    async def submit_transaction(self, m: SubmitTransactionIn) -> SubmitTransactionOut:
        """Tier 3 never executes: it records a PENDING_APPROVAL row and freezes (design Gate 7)."""
        self._need_tenant(m.tenant_id)
        aid = str(uuid.uuid4())
        async with self._lock:
            h = self._append(m.tenant_id, "tx.submitted", m.user_id, {
                "approval_id": aid, "action": m.action, "account_sha256": hashlib.sha256(m.account_id.encode()).hexdigest(),
                "amount": m.amount, "currency": m.currency})
            self.approvals.setdefault(m.tenant_id, {})[aid] = {
                "approval_id": aid, "status": "PENDING_APPROVAL", "action": m.action, "account_id": m.account_id,
                "amount": m.amount, "currency": m.currency, "requested_by": m.user_id, "justification": m.justification,
                "created_at": _now(), "decided_by": None}
        return SubmitTransactionOut(status="PENDING_APPROVAL", approval_id=aid, audit_hash=h)

    async def get_approval(self, tenant_id: str, approval_id: str) -> dict[str, Any] | None:
        self._need_tenant(tenant_id)
        return self.approvals.get(tenant_id, {}).get(approval_id)

    # ---- human decision (workbench HITL queue). Separation of duties: the requester can never approve their own request.
    async def pending(self, tenant_id: str) -> list[dict[str, Any]]:
        self._need_tenant(tenant_id)
        return [a for a in self.approvals.get(tenant_id, {}).values() if a["status"] == "PENDING_APPROVAL"]

    async def decide(self, tenant_id: str, approval_id: str, approve: bool, approver: str) -> dict[str, Any]:
        self._need_tenant(tenant_id)
        async with self._lock:
            row = self.approvals.get(tenant_id, {}).get(approval_id)
            if row is None:
                raise KeyError("approval not found")
            if row["status"] != "PENDING_APPROVAL":
                raise ValueError(f"already {row['status']}")
            if approver == row["requested_by"]:
                raise PermissionError("requester cannot approve their own request")
            row["status"], row["decided_by"] = ("APPROVED" if approve else "REJECTED"), approver
            self._append(tenant_id, "tx.approved" if approve else "tx.rejected", approver, {"approval_id": approval_id})
            return dict(row)


@contextmanager
def local_services() -> Iterator[InMemoryStateStore]:
    """Patch the two network dependencies that build_runtime() creates. The store and fake redis are shared for the process."""
    store, redis = InMemoryStateStore(), FakeRedis()

    async def create(*a: Any, **k: Any) -> InMemoryStateStore:
        return store

    with mock.patch("app.runtime.StateStore.create", create), mock.patch("redis.asyncio.from_url", lambda *a, **k: redis):
        yield store


async def build_local_runtime(s: Settings) -> Runtime:
    with local_services() as store:
        rt = await build_runtime(s)
    rt.store = store
    return rt

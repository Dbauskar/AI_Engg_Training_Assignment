"""Postgres schema + migration (docs/design.md v1.2, Section 3.4).

Isolation is enforced twice:
  1. every query in app/state_store.py filters on tenant_id explicitly, and
  2. row-level security (FORCE) makes the database itself refuse cross-tenant rows.
The application connects as a limited role (NOT a superuser, so RLS cannot be bypassed) that can only
SELECT/INSERT: it cannot UPDATE or DELETE audit rows, and it cannot approve anything (no UPDATE on approvals).
"""
from __future__ import annotations

import psycopg
from psycopg import sql

TABLES = ("session_facts", "audit_log", "approvals")

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_facts (
    tenant_id        text        NOT NULL CHECK (tenant_id ~ '^[a-z0-9][a-z0-9_-]{2,39}$'),
    session_id       uuid        NOT NULL,
    fact_key         text        NOT NULL CHECK (fact_key ~ '^[a-z][a-z0-9_.]{2,63}$'),
    value            text        NOT NULL CHECK (length(value) BETWEEN 1 AND 1000),
    verified         boolean     NOT NULL CHECK (verified),
    source_chunk_ids text[]      NOT NULL CHECK (cardinality(source_chunk_ids) BETWEEN 1 AND 8),
    created_at       text        NOT NULL,
    PRIMARY KEY (tenant_id, session_id, fact_key)
);

CREATE TABLE IF NOT EXISTS audit_log (
    tenant_id  text   NOT NULL CHECK (tenant_id ~ '^[a-z0-9][a-z0-9_-]{2,39}$'),
    seq        bigint NOT NULL CHECK (seq >= 1),
    event_type text   NOT NULL,
    actor      text   NOT NULL,
    payload    jsonb  NOT NULL,
    ts         text   NOT NULL,
    prev_hash  text   NOT NULL CHECK (prev_hash ~ '^[0-9a-f]{64}$'),
    hash       text   NOT NULL CHECK (hash ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (tenant_id, seq)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id   uuid        NOT NULL,
    tenant_id     text        NOT NULL CHECK (tenant_id ~ '^[a-z0-9][a-z0-9_-]{2,39}$'),
    request_id    uuid        NOT NULL,
    session_id    uuid        NOT NULL,
    requester     text        NOT NULL,
    action        text        NOT NULL CHECK (action IN ('TRANSFER','ORDER','STATUS_CHANGE')),
    account_id    text        NOT NULL CHECK (account_id ~ '^ACC-[0-9]{8}$'),
    amount        text        CHECK (amount ~ '^[0-9]{1,12}(\\.[0-9]{1,2})?$'),
    currency      text        CHECK (currency ~ '^[A-Z]{3}$'),
    justification text        NOT NULL,
    status        text        NOT NULL DEFAULT 'PENDING_APPROVAL'
                              CHECK (status IN ('PENDING_APPROVAL','APPROVED','REJECTED','EXPIRED')),
    approver      text,
    created_at    text        NOT NULL,
    PRIMARY KEY (tenant_id, approval_id),
    UNIQUE (tenant_id, request_id),
    CHECK (approver IS NULL OR approver <> requester)          -- separation of duties
);
"""


def migrate(admin_conninfo: str, app_user: str, app_password: str) -> None:
    """Idempotent: create tables, RLS policies, and the limited application role."""
    with psycopg.connect(admin_conninfo, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # reserved for later phases (pgvector)
        conn.execute(SCHEMA)
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (app_user,)).fetchone()
        role = sql.Identifier(app_user)
        if exists:
            conn.execute(sql.SQL("ALTER ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}")
                         .format(role, sql.Literal(app_password)))
        else:
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}")
                         .format(role, sql.Literal(app_password)))
        db = conn.execute("SELECT current_database()").fetchone()[0]
        conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(db), role))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        for t in TABLES:
            tbl = sql.Identifier(t)
            conn.execute(sql.SQL("REVOKE ALL ON {} FROM {}").format(tbl, role))
            # SELECT + INSERT only: append-only audit trail, and no path for the app to approve a request
            conn.execute(sql.SQL("GRANT SELECT, INSERT ON {} TO {}").format(tbl, role))
            conn.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(tbl))
            conn.execute(sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(tbl))
            conn.execute(sql.SQL("DROP POLICY IF EXISTS tenant_isolation ON {}").format(tbl))
            conn.execute(sql.SQL(
                "CREATE POLICY tenant_isolation ON {} "
                "USING (tenant_id = current_setting('app.tenant_id', true)) "
                "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))").format(tbl))
        # session_facts may be corrected by the same tenant (upsert), so it additionally needs UPDATE
        conn.execute(sql.SQL("GRANT UPDATE ON session_facts TO {}").format(role))

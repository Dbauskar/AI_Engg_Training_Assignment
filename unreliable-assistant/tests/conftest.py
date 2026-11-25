import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# The global tracer provider can only be set once per process.
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans():
    _exporter.clear()
    yield _exporter
    _exporter.clear()


# ---- throwaway Postgres database (created and dropped per test session) --------------------------------------
import os  # noqa: E402
import uuid  # noqa: E402

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

from app.config import load_env, pg_conninfo  # noqa: E402
from app.db import migrate  # noqa: E402


@pytest.fixture(scope="session")
def pg_test_db():
    load_env()
    try:
        server = psycopg.connect(pg_conninfo(admin=True, dbname="postgres"), autocommit=True, connect_timeout=3)
    except Exception:
        pytest.skip("Postgres not reachable on 127.0.0.1:5432")
    name = f"assistant_test_{uuid.uuid4().hex[:8]}"
    server.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        migrate(pg_conninfo(admin=True, dbname=name), os.environ.get("APP_DB_USER", "assistant_app"),
                os.environ["APP_DB_PASSWORD"])
        yield {"admin": pg_conninfo(admin=True, dbname=name), "app": pg_conninfo(dbname=name)}
    finally:
        server.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        server.close()

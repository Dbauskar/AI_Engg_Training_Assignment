"""Environment-driven configuration. Reads ./.env without overriding real environment variables."""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> None:
    p = path or _ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _req(name: str) -> str:
    load_env()
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"missing required setting {name} (see .env.example)")
    return v


def pg_conninfo(*, admin: bool = False, dbname: str | None = None) -> str:
    """libpq keyword string. `admin` is only for migrations/tests; the app connects as the limited role."""
    load_env()
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = dbname or os.environ.get("POSTGRES_DB", "assistant")
    if admin:
        user, pw = os.environ.get("POSTGRES_USER", "assistant"), _req("POSTGRES_PASSWORD")
    else:
        user, pw = os.environ.get("APP_DB_USER", "assistant_app"), _req("APP_DB_PASSWORD")
    return f"host={host} port={port} dbname={db} user={user} password={pw}"


def redis_url() -> str:
    load_env()
    return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

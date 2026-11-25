"""Create/upgrade the schema, RLS policies and the limited app role. Idempotent.

Usage: uv run python scripts/migrate.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import _req, load_env, pg_conninfo  # noqa: E402
from app.db import migrate  # noqa: E402

if __name__ == "__main__":
    load_env()
    migrate(pg_conninfo(admin=True), os.environ.get("APP_DB_USER", "assistant_app"), _req("APP_DB_PASSWORD"))
    print("migrated", os.environ.get("POSTGRES_DB", "assistant"))

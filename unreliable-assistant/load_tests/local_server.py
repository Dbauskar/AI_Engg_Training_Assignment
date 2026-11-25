"""Run the real API on the in-memory local stack (no Postgres/Redis/docker needed).
  uv run python -m load_tests.local_server [--port 8080] [--workers 1]
Numbers measured against this are LOCAL-STACK numbers: they say nothing about Postgres or Redis behaviour."""
import argparse

import uvicorn

from app.localstack import build_local_runtime
from app.main import create_app
from app.runtime import Settings


def make_app():
    return create_app(Settings.from_env(), runtime_factory=build_local_runtime)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    uvicorn.run(make_app(), host="127.0.0.1", port=a.port, log_level="warning")

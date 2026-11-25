"""Bulkhead isolation (docs/design.md Gate 9): separate capacity pools so one slow dependency cannot starve the rest.

Pools (design): rag=64, db=128, llm=256, plus a small triage pool so routing is never queued behind retrieval.
Each pool = a semaphore (admission control) + its own thread pool for blocking work. When a pool is full a caller
waits at most `queue_timeout_s`, then gets BulkheadFull (fail fast -> E_RATE_LIMITED) instead of queueing forever.
The event loop itself never runs blocking work, and /healthz never takes a slot, so health stays responsive.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, AsyncIterator, Callable


class BulkheadFull(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"bulkhead '{name}' is full")
        self.pool = name


class Bulkhead:
    def __init__(self, name: str, size: int, queue_timeout_s: float = 0.05) -> None:
        self.name, self.size, self.queue_timeout_s = name, size, queue_timeout_s
        self._sem = asyncio.Semaphore(size)
        self._exec = ThreadPoolExecutor(max_workers=size, thread_name_prefix=f"bh-{name}")
        self.in_flight = 0
        self.peak = 0
        self.rejected = 0
        self.completed = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self._sem.acquire(), self.queue_timeout_s)
        except asyncio.TimeoutError:
            self.rejected += 1
            raise BulkheadFull(self.name) from None
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            yield
        finally:
            self.in_flight -= 1
            self.completed += 1
            self._sem.release()

    async def run_sync(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run blocking `fn` on this pool's own threads (never on the event loop or another pool)."""
        async with self.slot():
            ctx = contextvars.copy_context()  # keeps the OTel trace context: pooled work stays inside the request's trace
            return await asyncio.get_running_loop().run_in_executor(self._exec, ctx.run, partial(fn, *args, **kwargs))

    def stats(self) -> dict[str, int]:
        return {"size": self.size, "in_flight": self.in_flight, "peak": self.peak,
                "rejected": self.rejected, "completed": self.completed}

    def shutdown(self) -> None:
        self._exec.shutdown(wait=False, cancel_futures=True)


@dataclass
class Pools:
    rag: Bulkhead
    db: Bulkhead
    llm: Bulkhead
    triage: Bulkhead

    @classmethod
    def from_env(cls, queue_timeout_s: float = 0.05) -> "Pools":
        g = lambda k, d: int(os.environ.get(k, d))  # noqa: E731
        return cls(Bulkhead("rag", g("RAG_POOL", 64), queue_timeout_s), Bulkhead("db", g("DB_POOL_SLOTS", 128), queue_timeout_s),
                   Bulkhead("llm", g("LLM_POOL", 256), queue_timeout_s), Bulkhead("triage", g("TRIAGE_POOL", 16), queue_timeout_s))

    def stats(self) -> dict[str, dict[str, int]]:
        return {p.name: p.stats() for p in (self.rag, self.db, self.llm, self.triage)}

    def shutdown(self) -> None:
        for p in (self.rag, self.db, self.llm, self.triage):
            p.shutdown()

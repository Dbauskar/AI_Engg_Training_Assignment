"""Liveness/health snapshot. Deliberately takes NO bulkhead slot, does no I/O and awaits nothing slow, so it stays
fast while every pool is saturated (design Gate 9). Prompt 10 exposes this as GET /healthz."""
from __future__ import annotations

import time

from app.bulkhead import Pools


async def health_snapshot(pools: Pools) -> dict:
    stats = pools.stats()
    saturated = [n for n, s in stats.items() if s["in_flight"] >= s["size"]]
    return {"status": "degraded" if saturated else "ok", "saturated_pools": saturated, "pools": stats,
            "ts": time.time()}

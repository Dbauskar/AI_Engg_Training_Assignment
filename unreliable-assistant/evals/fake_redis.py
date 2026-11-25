"""Minimal in-memory stand-in for redis.asyncio, only the commands app/cache.py and the idempotency store use.
It models NOTHING about Redis itself (no TTL expiry, no cluster slots): chaos results obtained with it prove the application's
own scoping/fallback logic, not Redis behaviour. evals/chaos.py uses a real Redis when one is reachable."""
from __future__ import annotations

import fnmatch
from typing import Any


class _Pipe:
    def __init__(self, r: "FakeRedis") -> None:
        self.r, self.ops = r, []

    def __getattr__(self, name: str):
        def add(*a: Any, **k: Any):
            self.ops.append((name, a, k))
            return self
        return add

    async def execute(self):
        return [await getattr(self.r, n)(*a, **k) for n, a, k in self.ops]


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}
        self.h: dict[str, dict] = {}
        self.z: dict[str, dict] = {}
        self.down = False

    def _chk(self) -> None:
        if self.down:
            from redis.exceptions import ConnectionError as RedisConnectionError
            raise RedisConnectionError("simulated outage")

    async def get(self, k): self._chk(); return self.kv.get(k)
    async def set(self, k, v, nx=False, ex=None):
        self._chk()
        if nx and k in self.kv:
            return None
        self.kv[k] = v
        return True
    async def delete(self, *ks): self._chk(); return sum(self.kv.pop(k, None) is not None for k in ks)
    async def hgetall(self, k): self._chk(); return dict(self.h.get(k, {}))
    async def hset(self, k, f, v): self._chk(); self.h.setdefault(k, {})[f] = v; return 1
    async def hdel(self, k, *fs): self._chk(); [self.h.get(k, {}).pop(f, None) for f in fs]; return len(fs)
    async def zadd(self, k, m): self._chk(); self.z.setdefault(k, {}).update(m); return len(m)
    async def zrem(self, k, *ms): self._chk(); [self.z.get(k, {}).pop(m, None) for m in ms]; return len(ms)
    async def zcard(self, k): self._chk(); return len(self.z.get(k, {}))
    async def zrange(self, k, a, b): self._chk(); items = sorted(self.z.get(k, {}), key=self.z.get(k, {}).get); return items[a:b + 1]
    async def expire(self, k, s): self._chk(); return True
    async def aclose(self) -> None:
        return None

    async def ping(self): self._chk(); return True
    def pipeline(self, transaction=False): return _Pipe(self)

    async def scan_iter(self, match="*", count=100):
        self._chk()
        for k in list(self.kv) + list(self.h) + list(self.z):
            if fnmatch.fnmatchcase(k, match):
                yield k

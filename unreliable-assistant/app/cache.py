"""Tenant-isolated Redis semantic cache (docs/design.md v1.2, ADR-3, FMEA #1, #6, #7).

Keys
  entry  cache:{<tenant>}:<sha256(tenant:user:normalised_query)>        JSON, TTL 300 s +/- 45 s jitter
  scope  cachesem:{<tenant>}:<sha256(tenant:user)>  (hash + zset)       bounded embedding index for that scope
The {tenant} hash tag keeps one tenant's keys in one Redis Cluster slot group.

Isolation: an exact hit needs the same tenant AND user; a semantic hit (cosine >= 0.92) is searched only inside the
caller's own tenant+user scope index, so another tenant's vectors are never even loaded. As a backstop every entry
carries its tenant/user and a mismatch is dropped, counted (leaks_blocked) and deleted.
Safety: only verified responses are cached (poisoning guard), the raw query is never stored (only its hash), Redis
failures degrade to a cache bypass, and `enabled=False` is the kill switch.
Scale note: scope search is a bounded in-process scan (max_scope_entries). A production deployment at high hit-rates
would swap this for RediSearch/vector sets behind the same interface.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

import numpy as np
import redis.asyncio as aioredis
from opentelemetry import trace
from redis.exceptions import RedisError

from app.contracts import TENANT_RE, USER_RE
from app.pii import PiiDetector, shared_detector

_tracer = trace.get_tracer("unreliable_assistant.cache")
_TENANT = re.compile(TENANT_RE)
_USER = re.compile(USER_RE)

TTL_S = 300
JITTER_S = 45
SIMILARITY = 0.92
MAX_SCOPE_ENTRIES = 64

Kind = Literal["exact", "semantic", "miss", "bypass"]
EmbedFn = Callable[[str], "np.ndarray"]
_LEAK = object()  # unique sentinel: can never collide with a real cached response


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?!. ")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def entry_key(tenant_id: str, user_id: str, query: str) -> str:
    return f"cache:{{{tenant_id}}}:{_sha(f'{tenant_id}:{user_id}:{normalize_query(query)}')}"


def scope_keys(tenant_id: str, user_id: str) -> tuple[str, str]:
    base = f"cachesem:{{{tenant_id}}}:{_sha(f'{tenant_id}:{user_id}')}"
    return base + ":vec", base + ":ord"


@dataclass
class CacheResult:
    kind: Kind
    response: str | None = None
    similarity: float | None = None

    @property
    def hit(self) -> bool:
        return self.kind in ("exact", "semantic")


@dataclass
class CacheStats:
    hits_exact: int = 0
    hits_semantic: int = 0
    misses: int = 0
    bypassed: int = 0
    errors: int = 0
    leaks_blocked: int = 0
    rejected_unverified: int = 0
    rejected_pii: int = 0
    puts: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits_exact + self.hits_semantic + self.misses
        return (self.hits_exact + self.hits_semantic) / total if total else 0.0


def _check_scope(tenant_id: str, user_id: str) -> None:
    if not isinstance(tenant_id, str) or not _TENANT.match(tenant_id):
        raise ValueError("tenant_id is required and must be well-formed")  # CLAUDE.md rule 3
    if not isinstance(user_id, str) or not _USER.match(user_id):
        raise ValueError("user_id is required and must be well-formed")


@dataclass
class SemanticCache:
    redis: aioredis.Redis
    embed: EmbedFn
    threshold: float = SIMILARITY
    ttl_s: int = TTL_S
    jitter_s: int = JITTER_S
    max_scope_entries: int = MAX_SCOPE_ENTRIES
    enabled: bool = True
    rng: random.Random = field(default_factory=random.Random)
    stats: CacheStats = field(default_factory=CacheStats)
    pii: PiiDetector | None = None  # persistence guard; defaults to the shared regex+NER detector

    # ---- helpers ------------------------------------------------------------------------------------------
    async def _embed(self, text: str) -> np.ndarray:
        v = await asyncio.to_thread(self.embed, text)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def _parse_entry(self, raw: bytes | str | None, tenant_id: str, user_id: str) -> Any:
        """Return the response only if the entry provably belongs to this tenant and user."""
        if raw is None:
            return None
        try:
            e = json.loads(raw)
            if e["tenant_id"] != tenant_id or e["user_id"] != user_id or e.get("verified") is not True:
                self.stats.leaks_blocked += 1
                return _LEAK
            return e["response"]
        except (ValueError, KeyError, TypeError):
            self.stats.errors += 1
            return None

    # ---- API ----------------------------------------------------------------------------------------------
    async def get(self, tenant_id: str, user_id: str, query: str) -> CacheResult:
        _check_scope(tenant_id, user_id)
        with _tracer.start_as_current_span("cache.lookup") as span:  # never records the query text
            span.set_attribute("tenant.id", tenant_id)
            res = await self._get(tenant_id, user_id, query)
            span.set_attribute("cache.result", res.kind)
            if res.similarity is not None:
                span.set_attribute("cache.similarity", res.similarity)
            span.set_attribute("cache.leaks_blocked_total", self.stats.leaks_blocked)
            return res

    async def _get(self, tenant_id: str, user_id: str, query: str) -> CacheResult:
        if not self.enabled:
            self.stats.bypassed += 1
            return CacheResult("bypass")
        try:
            key = entry_key(tenant_id, user_id, query)
            got = self._parse_entry(await self.redis.get(key), tenant_id, user_id)
            if got is _LEAK:
                await self.redis.delete(key)  # drop the offending entry so it cannot be served again
            elif got is not None:
                self.stats.hits_exact += 1
                return CacheResult("exact", got, 1.0)

            vec_key, ord_key = scope_keys(tenant_id, user_id)
            stored = await self.redis.hgetall(vec_key)
            if stored:
                q = await self._embed(query)
                ids = list(stored.keys())
                mat = np.vstack([np.frombuffer(stored[i], dtype=np.float32) for i in ids])
                sims = mat @ q
                for j in np.argsort(-sims)[:3]:
                    if sims[j] < self.threshold:
                        break
                    sha = ids[j].decode() if isinstance(ids[j], bytes) else ids[j]
                    ekey = f"cache:{{{tenant_id}}}:{sha}"
                    resp = self._parse_entry(await self.redis.get(ekey), tenant_id, user_id)
                    if resp is _LEAK:
                        await self.redis.delete(ekey)
                        continue
                    if resp is None:  # expired entry: tidy the index
                        await self.redis.hdel(vec_key, ids[j])
                        await self.redis.zrem(ord_key, ids[j])
                        continue
                    self.stats.hits_semantic += 1
                    return CacheResult("semantic", resp, float(sims[j]))
            self.stats.misses += 1
            return CacheResult("miss")
        except (RedisError, OSError, asyncio.TimeoutError):
            self.stats.errors += 1  # FMEA #1: Redis down -> bypass the cache, never fail the request
            return CacheResult("bypass")

    async def put(self, tenant_id: str, user_id: str, query: str, response: str, *, verified: bool,
                  source_hashes: tuple[str, ...] = ()) -> bool:
        """Cache a response. Only Verification-approved responses are accepted (FMEA #6)."""
        _check_scope(tenant_id, user_id)
        if not verified:
            self.stats.rejected_unverified += 1
            return False
        if not self.enabled:
            self.stats.bypassed += 1
            return False
        if await asyncio.to_thread((self.pii or shared_detector()).scan, response):  # zero raw PII persistence
            self.stats.rejected_pii += 1
            return False
        try:
            key = entry_key(tenant_id, user_id, query)
            sha = key.rsplit(":", 1)[1]
            ttl = max(1, round(self.ttl_s + self.rng.uniform(-self.jitter_s, self.jitter_s)))  # anti-stampede
            entry = json.dumps({
                "tenant_id": tenant_id, "user_id": user_id, "verified": True, "response": response,
                "source_hashes": list(source_hashes), "created": datetime.now(timezone.utc).isoformat(),
            })
            vec = (await self._embed(query)).astype(np.float32).tobytes()
            vec_key, ord_key = scope_keys(tenant_id, user_id)
            now = datetime.now(timezone.utc).timestamp()
            pipe = self.redis.pipeline(transaction=False)
            pipe.set(key, entry, ex=ttl)
            pipe.hset(vec_key, sha, vec)
            pipe.zadd(ord_key, {sha: now})
            pipe.expire(vec_key, ttl * 2)
            pipe.expire(ord_key, ttl * 2)
            await pipe.execute()
            excess = await self.redis.zcard(ord_key) - self.max_scope_entries
            if excess > 0:  # bounded index: evict the oldest
                old = await self.redis.zrange(ord_key, 0, excess - 1)
                if old:
                    await self.redis.zrem(ord_key, *old)
                    await self.redis.hdel(vec_key, *old)
            self.stats.puts += 1
            return True
        except (RedisError, OSError, asyncio.TimeoutError):
            self.stats.errors += 1
            return False

    async def purge_tenant(self, tenant_id: str) -> int:
        """Delete every cache key of one tenant (incident response for suspected poisoning)."""
        if not isinstance(tenant_id, str) or not _TENANT.match(tenant_id):
            raise ValueError("tenant_id is required and must be well-formed")
        n = 0
        for pattern in (f"cache:{{{tenant_id}}}:*", f"cachesem:{{{tenant_id}}}:*"):
            batch: list[Any] = []
            async for k in self.redis.scan_iter(match=pattern, count=500):
                batch.append(k)
                if len(batch) >= 500:
                    n += await self.redis.delete(*batch)
                    batch = []
            if batch:
                n += await self.redis.delete(*batch)
        return n

import asyncio
import json
import random
import uuid

import numpy as np
import pytest
import redis.asyncio as aioredis

from app.cache import SemanticCache, entry_key, normalize_query, scope_keys
from app.pii import PiiDetector

FAST_PII = PiiDetector(use_ner=False)  # volume tests: NER per put would dominate runtime

QUERY = "How many vacation days do I get each year?"


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def vec_with_cosine(cos):  # unit vector whose cosine with e0 is exactly `cos`
    return unit(cos, float(np.sqrt(1 - cos ** 2)), 0, 0)


E0 = unit(1, 0, 0, 0)
EMBEDDINGS = {  # query -> vector; everything unlisted is orthogonal
    normalize_query(QUERY): E0,
    normalize_query("How many holiday days do employees receive annually?"): vec_with_cosine(0.95),
    normalize_query("What is the notice period for resignation?"): vec_with_cosine(0.80),
}


def fake_embed(text: str) -> np.ndarray:
    return EMBEDDINGS.get(normalize_query(text), unit(0, 0, 1, 0))


@pytest.fixture
async def redis_client():
    c = aioredis.Redis(host="127.0.0.1", port=6379, socket_timeout=5, max_connections=300)  # > test concurrency (200)
    try:
        await c.ping()
    except Exception:
        pytest.skip("Redis not reachable on 127.0.0.1:6379")
    yield c
    await c.aclose()


@pytest.fixture
def tenants():
    t = uuid.uuid4().hex[:8]
    return f"ta-{t}", f"tb-{t}"


@pytest.fixture
async def cache(redis_client, tenants):
    c = SemanticCache(redis_client, fake_embed, pii=FAST_PII, rng=random.Random(7))
    yield c
    for t in tenants:
        await c.purge_tenant(t)


# ---------------- keys & storage hygiene ----------------
def test_key_format_and_normalisation():
    k = entry_key("acme-corp", "u1", "  How MANY days?  ")
    assert k.startswith("cache:{acme-corp}:") and len(k.rsplit(":", 1)[1]) == 64
    assert entry_key("acme-corp", "u1", "how many days") == k
    assert entry_key("other-co", "u1", "how many days") != k  # tenant is part of the hash
    assert entry_key("acme-corp", "u2", "how many days") != k  # so is the user
    assert scope_keys("acme-corp", "u1")[0].startswith("cachesem:{acme-corp}:")


async def test_requires_valid_tenant_and_user(cache):
    for bad in (None, "", "UPPER", "a/b", "x"):
        with pytest.raises(ValueError):
            await cache.get(bad, "u1", QUERY)
        with pytest.raises(ValueError):
            await cache.put(bad, "u1", QUERY, "r", verified=True)
    with pytest.raises(ValueError):
        await cache.get("acme-corp", "bad user!", QUERY)


async def test_raw_query_text_is_never_stored_in_redis(cache, redis_client, tenants):
    a, _ = tenants
    secret_q = "my SSN is 123-45-6789, what is my leave balance"
    await cache.put(a, "u1", secret_q, "answer", verified=True)
    keys = [k async for k in redis_client.scan_iter(match=f"*{{{a}}}*")]
    assert keys  # something was stored
    for k in keys:
        t = (await redis_client.type(k)).decode()
        raw = {"string": redis_client.get, "hash": redis_client.hgetall,
               "zset": lambda key: redis_client.zrange(key, 0, -1, withscores=True)}[t]
        blob = repr(await raw(k)) + k.decode()
        assert "123-45-6789" not in blob and "leave balance" not in blob


# ---------------- hits and misses ----------------
async def test_exact_hit_miss_and_scoping(cache, tenants):
    a, b = tenants
    assert (await cache.get(a, "u1", QUERY)).kind == "miss"
    assert await cache.put(a, "u1", QUERY, "25 days", verified=True)
    r = await cache.get(a, "u1", "  how many VACATION days do i get each year  ")  # normalises to the same key
    assert r.kind == "exact" and r.response == "25 days"
    assert (await cache.get(a, "u2", QUERY)).kind == "miss"  # same tenant, other user
    assert (await cache.get(b, "u1", QUERY)).kind == "miss"  # other tenant, same user


async def test_semantic_hit_at_or_above_threshold_only(cache, tenants):
    a, _ = tenants
    await cache.put(a, "u1", QUERY, "25 days", verified=True)
    hit = await cache.get(a, "u1", "How many holiday days do employees receive annually?")  # cosine 0.95
    assert hit.kind == "semantic" and hit.response == "25 days" and hit.similarity == pytest.approx(0.95, abs=1e-4)
    miss = await cache.get(a, "u1", "What is the notice period for resignation?")  # cosine 0.80 < 0.92
    assert miss.kind == "miss"


async def test_semantic_search_never_crosses_tenant_or_user(cache, tenants):
    a, b = tenants
    await cache.put(a, "u1", QUERY, "A's answer", verified=True)
    paraphrase = "How many holiday days do employees receive annually?"
    assert (await cache.get(b, "u1", paraphrase)).kind == "miss"
    assert (await cache.get(a, "u2", paraphrase)).kind == "miss"
    assert (await cache.get(a, "u1", paraphrase)).kind == "semantic"


# ---------------- safety rules ----------------
async def test_unverified_responses_are_not_cached(cache, tenants):
    a, _ = tenants
    assert await cache.put(a, "u1", QUERY, "maybe wrong", verified=False) is False
    assert (await cache.get(a, "u1", QUERY)).kind == "miss"
    assert cache.stats.rejected_unverified == 1


async def test_ttl_has_jitter_within_bounds(cache, redis_client, tenants):
    a, _ = tenants
    ttls = set()
    for i in range(40):
        await cache.put(a, "u1", f"query number {i}", "r", verified=True)
        ttls.add(await redis_client.ttl(entry_key(a, "u1", f"query number {i}")))
    assert min(ttls) >= 254 and max(ttls) <= 345  # 300 +/- 45 (allow 1 s elapsed)
    assert len(ttls) > 10  # actually jittered, so entries do not expire in lock-step


async def test_scope_index_is_bounded(redis_client, tenants):
    a, _ = tenants
    c = SemanticCache(redis_client, fake_embed, pii=FAST_PII, max_scope_entries=10)
    try:
        for i in range(40):
            await c.put(a, "u1", f"distinct query {i}", "r", verified=True)
        vec_key, ord_key = scope_keys(a, "u1")
        assert await redis_client.hlen(vec_key) <= 10 and await redis_client.zcard(ord_key) <= 10
    finally:
        await c.purge_tenant(a)


async def test_colliding_or_poisoned_entry_is_blocked_and_removed(cache, redis_client, tenants):
    a, b = tenants
    # simulate a key collision / corruption: tenant B's entry sitting under tenant A's key
    key = entry_key(a, "u1", QUERY)
    await redis_client.set(key, json.dumps({"tenant_id": b, "user_id": "u1", "verified": True, "response": "B-SECRET"}), ex=60)
    r = await cache.get(a, "u1", QUERY)
    assert r.kind == "miss" and r.response is None
    assert cache.stats.leaks_blocked == 1
    assert await redis_client.exists(key) == 0  # offending entry deleted
    # an unverified entry planted directly is refused too
    await redis_client.set(key, json.dumps({"tenant_id": a, "user_id": "u1", "verified": False, "response": "x"}), ex=60)
    assert (await cache.get(a, "u1", QUERY)).response is None


async def test_response_equal_to_internal_marker_is_a_normal_hit(cache, tenants):
    a, _ = tenants
    await cache.put(a, "u1", QUERY, "__LEAK__", verified=True)
    r = await cache.get(a, "u1", QUERY)
    assert r.kind == "exact" and r.response == "__LEAK__"


async def test_kill_switch_bypasses_cache(redis_client, tenants):
    a, _ = tenants
    c = SemanticCache(redis_client, fake_embed, pii=FAST_PII, enabled=False)
    assert await c.put(a, "u1", QUERY, "r", verified=True) is False
    assert (await c.get(a, "u1", QUERY)).kind == "bypass"


async def test_redis_down_degrades_to_bypass_not_error(tenants):
    a, _ = tenants
    dead = aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, socket_timeout=0.2)
    c = SemanticCache(dead, fake_embed, pii=FAST_PII)
    assert (await c.get(a, "u1", QUERY)).kind == "bypass"
    assert await c.put(a, "u1", QUERY, "r", verified=True) is False
    assert c.stats.errors >= 2
    await dead.aclose()


async def test_purge_tenant_only_touches_that_tenant(cache, tenants):
    a, b = tenants
    await cache.put(a, "u1", QUERY, "A", verified=True)
    await cache.put(b, "u1", QUERY, "B", verified=True)
    assert await cache.purge_tenant(a) >= 2
    assert (await cache.get(a, "u1", QUERY)).kind == "miss"
    assert (await cache.get(b, "u1", QUERY)).response == "B"


# ---------------- observability ----------------
async def test_lookup_span_records_result_but_never_the_query(cache, tenants, spans):
    a, _ = tenants
    q = "my email is alice@example.com, what is my leave balance"
    await cache.put(a, "u1", q, "r", verified=True)
    await cache.get(a, "u1", q)
    sp = [s for s in spans.get_finished_spans() if s.name == "cache.lookup"]
    assert sp and sp[-1].attributes["cache.result"] == "exact" and sp[-1].attributes["tenant.id"] == a
    assert "alice@example.com" not in " ".join(repr(dict(s.attributes)) for s in spans.get_finished_spans())


# ---------------- the required concurrency test ----------------
async def test_two_tenants_same_query_1000x_in_parallel_zero_cross_tenant_hits(redis_client, tenants):
    """Two tenants hammer the SAME user id and SAME query 1,000 times each, concurrently.
    Half the reads use a paraphrase to force the semantic path. No read may ever see the other tenant's answer."""
    a, b = tenants
    c = SemanticCache(redis_client, fake_embed, pii=FAST_PII, rng=random.Random(1))
    paraphrase = "How many holiday days do employees receive annually?"
    answers = {a: "ANSWER-FOR-TENANT-A", b: "ANSWER-FOR-TENANT-B"}
    observed: list[tuple[str, str | None]] = []
    sem = asyncio.Semaphore(200)

    async def writer(t):
        async with sem:
            assert await c.put(t, "same-user", QUERY, answers[t], verified=True)

    async def reader(t, i):
        async with sem:
            r = await c.get(t, "same-user", paraphrase if i % 2 else QUERY)
            observed.append((t, r.response))

    try:
        await c.put(a, "same-user", QUERY, answers[a], verified=True)  # seed so reads can hit
        await c.put(b, "same-user", QUERY, answers[b], verified=True)
        tasks = []
        for i in range(1000):
            tasks += [writer(a), writer(b), reader(a, i), reader(b, i)]
        random.Random(3).shuffle(tasks)
        await asyncio.gather(*tasks)

        assert len(observed) == 2000
        cross = [(t, r) for t, r in observed if r is not None and r != answers[t]]
        assert cross == [], f"cross-tenant responses served: {cross[:3]}"
        assert c.stats.errors == 0  # no silent degrade-to-bypass hiding a weak pass
        assert c.stats.leaks_blocked == 0  # isolation held by the keys themselves, not just the backstop
        hits = [r for t, r in observed if r is not None]
        assert len(hits) > 1900  # and the cache really was serving (this is not a trivial all-miss pass)
        assert c.stats.hits_exact > 0 and c.stats.hits_semantic > 0
    finally:
        for t in (a, b):
            await c.purge_tenant(t)

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

from agents.react import Act, Final
from agents.subagents import AnswerResult
from app.auth import TokenError, mint_token, verify_token
from app.bulkhead import Bulkhead, Pools
from app.harness import Tier
from app.main import create_app
from app.runtime import Runtime, Settings
from tests.test_plugins import make_pipeline
from tests.test_subagents import CHUNK, GOOD, SUBMIT, World

SECRET = b"k" * 40


def pools(rag=4):
    return Pools(Bulkhead("rag", rag), Bulkhead("db", 8), Bulkhead("llm", 8), Bulkhead("triage", 8))


def fake_runtime(route="HYBRID_RAG", script=None, probes=None, rag=4):
    w = World()
    pipe, reg = make_pipeline(w, route, script=script)
    rt = Runtime(Settings(token_secret=SECRET), pools(rag), pipe, reg, None, w.h, probes=probes or {})
    rt.world = w
    return rt


@asynccontextmanager
async def client_for(rt):
    async def factory(_):
        return rt

    app = create_app(Settings(token_secret=SECRET), runtime_factory=factory)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            yield c


def auth(tier=Tier.TIER_1_READ, user="alice", tenant="acme-corp", ttl=3600, secret=SECRET):
    return {"Authorization": "Bearer " + mint_token(secret, user, tenant, tier, ttl)}


Q = "Who has to approve a wire transfer above 10,000 USD?"


# =============================== tokens ===============================
def test_token_roundtrip_and_rejections():
    t = mint_token(SECRET, "alice", "acme-corp", Tier.TIER_2_COMPUTE, 60, now=1000)
    c = verify_token(SECRET, t, now=1010)
    assert (c.user_id, c.tenant_id, c.max_tier) == ("alice", "acme-corp", Tier.TIER_2_COMPUTE)
    with pytest.raises(TokenError):
        verify_token(SECRET, t, now=1061)  # expired
    with pytest.raises(TokenError):
        verify_token(b"x" * 40, t, now=1010)  # wrong secret
    head, body, sig = t.split(".")
    import base64
    forged = base64.urlsafe_b64encode(json.dumps({"sub": "alice", "tid": "other-co", "tier": 3, "iat": 1000, "exp": 9999999999, "v": 1}).encode()).rstrip(b"=").decode()
    for bad in (f"{head}.{forged}.{sig}", t[:-3] + "AAA", "garbage", "", "a.b.c", "ua1.e30.AAAA"):
        with pytest.raises(TokenError):
            verify_token(SECRET, bad, now=1010)
    with pytest.raises(ValueError):
        mint_token(b"short", "alice", "acme-corp")
    with pytest.raises(ValueError):
        mint_token(SECRET, "alice", "BAD TENANT")


# =============================== auth & validation ===============================
async def test_requests_without_valid_credentials_are_rejected_with_401():
    async with client_for(fake_runtime()) as c:
        body = {"query": Q}
        r = await c.post("/v1/answer", json=body)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert (await c.post("/v1/answer", json=body, headers={"Authorization": "Bearer nonsense"})).status_code == 401
        assert (await c.post("/v1/answer", json=body, headers=auth(secret=b"z" * 40))).status_code == 401
        assert (await c.post("/v1/answer", json=body, headers=auth(ttl=-5))).status_code == 401
        assert (await c.post("/v1/answer", json=body, headers={"Authorization": "Basic abc"})).status_code == 401
        assert "tenant" not in r.text.lower()  # no hints about why


async def test_tenant_user_and_tier_come_only_from_the_token_never_from_the_body():
    rt = fake_runtime()
    seen = []
    orig = rt.pipeline.answer

    async def spy(caller, q, s):
        seen.append(caller)
        return await orig(caller, q, s)

    rt.pipeline.answer = spy
    async with client_for(rt) as c:
        r = await c.post("/v1/answer", json={"query": Q}, headers=auth(Tier.TIER_2_COMPUTE, user="bob", tenant="globex-inc"))
        assert r.status_code == 200
        r2 = await c.post("/v1/answer", json={"query": Q, "tenant_id": "acme-corp", "tier": 3}, headers=auth())
        assert r2.status_code == 422  # unknown fields are refused, so the body cannot even try to claim an identity
    assert (seen[0].tenant_id, seen[0].user_id, seen[0].max_tier) == ("globex-inc", "bob", Tier.TIER_2_COMPUTE) and len(seen) == 1


async def test_the_token_tier_is_the_ceiling_for_state_changing_requests():
    args = dict(SUBMIT)
    script = [Act("submit_transaction", args)]
    for tier, expected in ((Tier.TIER_1_READ, "HALTED"), (Tier.TIER_3_MUTATE, "FROZEN")):
        rt = fake_runtime("TRANSACTION_EXEC", script)
        async with client_for(rt) as c:
            r = await c.post("/v1/answer", json={"query": "Transfer 100.50 USD to ACC-12345678"}, headers=auth(tier))
        assert r.json()["outcome"] == expected, (tier, r.json())
        assert rt.world.submit_calls == (1 if expected == "FROZEN" else 0)


@pytest.mark.parametrize("body", [{}, {"query": ""}, {"query": "x" * 2001}, {"query": "ok", "session_id": "not-a-uuid"}, {"query": "ok", "stream": "maybe"},
                                  {"query": 5}])
async def test_invalid_bodies_are_rejected_before_any_work(body):
    rt = fake_runtime()
    async with client_for(rt) as c:
        r = await c.post("/v1/answer", json=body, headers=auth())
    assert r.status_code == 422 and rt.world.retrieve_calls == 0


async def test_oversized_bodies_get_413_and_internals_are_not_documented():
    async with client_for(fake_runtime()) as c:
        r = await c.post("/v1/answer", content=b"x" * 20_000, headers={**auth(), "content-type": "application/json"})
        assert r.status_code == 413
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await c.get(path)).status_code == 404


# =============================== responses ===============================
async def test_json_response_shape_and_trace_headers():
    async with client_for(fake_runtime()) as c:
        r = await c.post("/v1/answer", json={"query": Q}, headers=auth())
    j = r.json()
    assert r.status_code == 200 and j["outcome"] == "ANSWERED" and j["route"] == "HYBRID_RAG" and j["answer"] == GOOD
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"]) and j["request_id"] == r.headers["x-request-id"]
    m = re.fullmatch(r"00-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}", r.headers["traceparent"])
    assert m, r.headers["traceparent"]
    assert j["trace_id"] == m.group(1)
    assert uuid.UUID(j["session_id"]).version == 4 and j["timings_ms"]["total"] > 0 and "hook.pre" in j["timings_ms"]


async def test_a_client_supplied_session_id_is_echoed():
    sid = str(uuid.uuid4())
    async with client_for(fake_runtime()) as c:
        r = await c.post("/v1/answer", json={"query": Q, "session_id": sid}, headers=auth())
    assert r.json()["session_id"] == sid


async def test_incoming_traceparent_is_continued_and_garbage_starts_a_fresh_trace(spans):
    tid = "4bf92f3577b34da6a3ce929d0e0e4736"
    async with client_for(fake_runtime()) as c:
        r = await c.post("/v1/answer", json={"query": Q}, headers={**auth(), "traceparent": f"00-{tid}-00f067aa0ba902b7-01"})
        assert r.headers["traceparent"].split("-")[1] == tid and r.json()["trace_id"] == tid
        mine = {s.name for s in spans.get_finished_spans() if format(s.context.trace_id, "032x") == tid}
        assert {"http.request", "hook.pre", "hook.post", "plugin.pii_redactor", "agent.react.step", "orchestrator.answer"} <= mine
        root = next(s for s in spans.get_finished_spans() if s.name == "http.request")
        assert root.parent is not None and format(root.parent.span_id, "016x") == "00f067aa0ba902b7"  # child of the caller's span
        r = await c.post("/v1/answer", json={"query": Q}, headers={**auth(), "traceparent": "not-a-traceparent"})
        assert r.status_code == 200 and r.json()["trace_id"] != tid and len(r.json()["trace_id"]) == 32


async def test_spans_never_contain_the_query_or_the_token(spans):
    token_headers = auth()
    async with client_for(fake_runtime()) as c:
        await c.post("/v1/answer", json={"query": "my email is jane_doe[at]example.org, who approves wires?"}, headers=token_headers)
    dump = " ".join(repr(dict(s.attributes)) + s.name for s in spans.get_finished_spans())
    assert "jane_doe" not in dump and token_headers["Authorization"].split()[1] not in dump


# =============================== streaming ===============================
def parse_sse(text):
    return [(m.group(1), json.loads(m.group(2))) for m in re.finditer(r"event: (\w+)\ndata: (.*)\n\n", text)]


async def test_sse_stream_order_and_content():
    async with client_for(fake_runtime()) as c:
        r = await c.post("/v1/answer", json={"query": Q, "stream": True}, headers=auth())
    assert r.headers["content-type"].startswith("text/event-stream") and r.headers["cache-control"] == "no-store"
    ev = parse_sse(r.text)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "accepted" and kinds[1] == "meta" and kinds[-1] == "done" and set(kinds[2:-1]) == {"token"}
    assert "".join(d["text"] for k, d in ev if k == "token") == GOOD
    assert ev[1][1]["outcome"] == "ANSWERED" and ev[-1][1]["http_equivalent"] == 200 and re.fullmatch(r"[0-9a-f]{32}", ev[-1][1]["trace_id"])


async def test_sse_stream_reports_failures_in_band_without_leaking_details():
    rt = fake_runtime()

    async def boom(*a, **k):
        raise RuntimeError("db password is hunter2")

    rt.pipeline.answer = boom
    async with client_for(rt) as c:
        r = await c.post("/v1/answer", json={"query": Q, "stream": True}, headers=auth())
    ev = parse_sse(r.text)
    assert ev[-1][0] == "error" and "hunter2" not in r.text


# =============================== error mapping ===============================
async def test_overload_maps_to_429_with_retry_after_and_degraded_to_503():
    rt = fake_runtime()
    for result, status in ((AnswerResult("busy", "HYBRID_RAG", "DEGRADED", reason="E_RATE_LIMITED"), 429),
                           (AnswerResult("down", "HYBRID_RAG", "DEGRADED", reason="E_UPSTREAM_UNAVAILABLE"), 503),
                           (AnswerResult("busy", "HYBRID_RAG", "HALTED", halt_reason="OVERLOADED"), 429)):
        async def fixed(*a, _r=result, **k):
            return _r

        rt.pipeline.answer = fixed
        async with client_for(rt) as c:
            r = await c.post("/v1/answer", json={"query": Q}, headers=auth())
        assert r.status_code == status and (r.headers.get("retry-after") == "1") == (status == 429)


async def test_unexpected_errors_return_a_generic_500_with_a_trace_id():
    rt = fake_runtime()

    async def boom(*a, **k):
        raise RuntimeError("secret connection string postgres://u:hunter2@db")

    rt.pipeline.answer = boom
    async with client_for(rt) as c:
        r = await c.post("/v1/answer", json={"query": Q}, headers=auth())
    assert r.status_code == 500 and r.json()["error"] == "internal_error" and len(r.json()["trace_id"]) == 32 and "hunter2" not in r.text
    assert "traceparent" in r.headers


# =============================== health ===============================
async def test_healthz_stays_fast_while_pools_are_saturated_and_requests_are_in_flight():
    rt = fake_runtime(rag=2)
    slow = asyncio.Event()
    orig = rt.pipeline.answer

    async def slow_answer(caller, q, s):
        await slow.wait()
        return await orig(caller, q, s)

    rt.pipeline.answer = slow_answer
    async with client_for(rt) as c:
        inflight = [asyncio.create_task(c.post("/v1/answer", json={"query": Q}, headers=auth())) for _ in range(6)]
        hold = [asyncio.create_task(rt.pools.rag.run_sync(time.sleep, 0.8)) for _ in range(2)]  # the rag pool is now full
        await asyncio.sleep(0.15)
        lat = []
        for _ in range(10):
            t = time.perf_counter()
            r = await c.get("/healthz")
            lat.append((time.perf_counter() - t) * 1000)
        body = r.json()
        slow.set()
        await asyncio.gather(*inflight, *hold)
    assert r.status_code == 200 and body["status"] == "degraded" and "rag" in body["saturated_pools"]
    assert max(lat) < 50 and sorted(lat)[len(lat) // 2] < 20, f"healthz latencies ms: {[round(x, 1) for x in lat]}"


async def test_readyz_reflects_dependency_health_and_warmup():
    state = {"pg": True}

    async def pg():
        return state["pg"]

    async def redis_down():
        raise ConnectionError("down")

    rt = fake_runtime(probes={"postgres": pg})
    async with client_for(rt) as c:
        ok = await c.get("/readyz")
        assert ok.status_code == 200 and ok.json()["dependencies"] == {"postgres": True}
        state["pg"] = False
        assert (await c.get("/readyz")).status_code == 503
        assert (await c.get("/healthz")).status_code == 200  # liveness is independent of dependencies
    rt2 = fake_runtime(probes={"redis": redis_down})
    async with client_for(rt2) as c:
        r = await c.get("/readyz")
        assert r.status_code == 503 and r.json()["dependencies"] == {"redis": False}

"""FastAPI entry point: ingress -> auth -> pre-plugins -> cache -> triage -> subagents -> post-plugins (docs/design.md 3.2).

Run:  uv run uvicorn app.main:app --host 127.0.0.1 --port 8080        (needs .env; see .env.example)

Endpoints
  POST /v1/answer   JSON, or Server-Sent Events when "stream": true.  Authorization: Bearer <token> (scripts/mint_token.py)
  GET  /healthz     liveness. Takes no bulkhead slot and does no I/O, so it answers while every pool is saturated.
  GET  /readyz      readiness: warm-up done and Postgres + Redis reachable (503 otherwise).

Tracing: an incoming W3C `traceparent` is continued (or a fresh trace started); the response carries `traceparent` and
`x-request-id`. No query text, answer text or token is ever put in a span attribute or a log line.

Streaming caveat: there is no token-level LLM stream yet (the answerer is deterministic), so the answer is produced whole and
then sent in small chunks. The first SSE event arrives immediately, but the first ANSWER chunk still waits for the pipeline.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import metrics, trace
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from typing import Annotated

from agents.subagents import AnswerResult
from app.auth import TokenError, verify_token
from app.bulkhead import BulkheadFull
from app.contracts import UUID4
from app.harness import Caller
from app.health import health_snapshot
from app.runtime import Runtime, Settings, build_runtime
from app.telemetry import register_runtime_metrics, setup_telemetry

_tracer = trace.get_tracer("unreliable_assistant.http")
_propagator = TraceContextTextMapPropagator()


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    session_id: UUID4 | None = None
    stream: bool = False


class AnswerResponse(BaseModel):
    request_id: str
    trace_id: str
    session_id: str
    answer: str
    outcome: str
    route: str
    abstained: bool
    cache_hit: bool
    timings_ms: dict[str, float]


class TraceMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware) so the server span stays open for the whole response, streaming included."""

    def __init__(self, app: ASGIApp, max_body_bytes: int = 16_384) -> None:
        self.app, self.max_body = app, max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        carrier = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        parent = _propagator.extract(carrier)  # invalid or missing traceparent -> a fresh root trace
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        with _tracer.start_as_current_span("http.request", context=parent, kind=SpanKind.SERVER, attributes={
                "http.method": scope["method"], "http.route": scope["path"], "request.id": request_id}) as span:
            if int(carrier.get("content-length", "0") or 0) > self.max_body:
                span.set_attribute("http.status_code", 413)
                return await _plain(send, 413, request_id, span, b'{"error":"request body too large"}')

            async def send_wrapper(message: Message) -> None:
                if message["type"] == "http.response.start":
                    span.set_attribute("http.status_code", message["status"])
                    if message["status"] >= 500:
                        span.set_status(Status(StatusCode.ERROR))
                    out: dict[str, str] = {}
                    inject(out)
                    h = MutableHeaders(scope=message)
                    h["x-request-id"] = request_id
                    if "traceparent" in out:
                        h["traceparent"] = out["traceparent"]
                await send(message)

            await self.app(scope, receive, send_wrapper)


async def _plain(send: Send, status: int, request_id: str, span: Any, body: bytes) -> None:
    out: dict[str, str] = {}
    inject(out)
    headers = [(b"content-type", b"application/json"), (b"x-request-id", request_id.encode())]
    if "traceparent" in out:
        headers.append((b"traceparent", out["traceparent"].encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""


def _status_for(res: AnswerResult) -> int:
    if res.reason == "E_RATE_LIMITED" or res.halt_reason == "OVERLOADED":
        return 429
    return 503 if res.outcome == "DEGRADED" else 200


def _to_response(res: AnswerResult, session_id: str, request_id: str) -> dict[str, Any]:
    return AnswerResponse(request_id=request_id, trace_id=_trace_id(), session_id=session_id, answer=res.answer, outcome=res.outcome,
                          route=res.route, abstained=res.abstained, cache_hit=res.cache_hit,
                          timings_ms={k: round(v, 1) for k, v in res.timings_ms.items()}).model_dump()


def _chunks(text: str, words: int = 4) -> list[str]:
    parts = re.findall(r"\S+\s*", text)
    return ["".join(parts[i:i + words]) for i in range(0, len(parts), words)]


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def create_app(settings: Settings | None = None, runtime_factory: Callable[[Settings], Awaitable[Runtime]] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        s = settings or Settings.from_env()
        if s.otlp_endpoint:
            setup_telemetry(s.service_name, s.otlp_endpoint)
        rt = await (runtime_factory or build_runtime)(s)
        if s.otlp_endpoint:
            register_runtime_metrics(metrics.get_meter("unreliable_assistant"), rt.pools, rt.cache)
        app.state.runtime = rt
        await rt.warmup()
        yield
        await rt.close()

    app = FastAPI(title="unreliable-assistant", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TraceMiddleware)

    def runtime(request: Request) -> Runtime:
        return request.app.state.runtime

    async def authenticate(request: Request) -> Caller:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "authentication required", headers={"WWW-Authenticate": "Bearer"})
        try:
            c = verify_token(runtime(request).settings.token_secret, token.strip())
        except TokenError:
            raise HTTPException(401, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"}) from None
        trace.get_current_span().set_attribute("tenant.id", c.tenant_id)  # tenant id is not sensitive; the user id is left out
        return Caller(c.tenant_id, c.user_id, c.max_tier)

    async def run(rt: Runtime, caller: Caller, body: AnswerRequest, session_id: str) -> AnswerResult:
        try:
            return await rt.pipeline.answer(caller, body.query, session_id)
        except BulkheadFull:
            return AnswerResult("The system is at capacity; please retry shortly.", "UNKNOWN", "DEGRADED", reason="E_RATE_LIMITED")
        except Exception as e:  # noqa: BLE001 - never leak internals or input echoes
            trace.get_current_span().set_attribute("error.type", type(e).__name__)
            raise

    @app.post("/v1/answer")
    async def answer(body: AnswerRequest, request: Request, caller: Caller = Depends(authenticate)):
        rt = runtime(request)
        session_id = body.session_id or str(uuid.uuid4())
        request_id = request.scope.get("state", {}).get("request_id", "")
        if body.stream:
            async def events() -> AsyncIterator[bytes]:
                yield _sse("accepted", {"session_id": session_id})  # first byte now; the answer follows when the pipeline completes
                try:
                    res = await run(rt, caller, body, session_id)
                except Exception:  # noqa: BLE001
                    yield _sse("error", {"error": "internal_error", "trace_id": _trace_id()})
                    return
                yield _sse("meta", {"route": res.route, "outcome": res.outcome, "abstained": res.abstained, "cache_hit": res.cache_hit})
                for piece in _chunks(res.answer):
                    yield _sse("token", {"text": piece})
                    await asyncio.sleep(0)  # let the server flush each chunk
                yield _sse("done", {"http_equivalent": _status_for(res), "trace_id": _trace_id(), "timings_ms": {k: round(v, 1) for k, v in res.timings_ms.items()}})

            return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        try:
            res = await run(rt, caller, body, session_id)
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "internal_error", "trace_id": _trace_id()}, status_code=500)
        status = _status_for(res)
        return JSONResponse(_to_response(res, session_id, request_id), status_code=status, headers={"Retry-After": "1"} if status == 429 else None)

    @app.get("/healthz")
    async def healthz(request: Request):
        # liveness only: no bulkhead slot, no database/redis call, nothing awaited that can block (design Gate 9)
        rt = runtime(request)
        snap = await health_snapshot(rt.pools)
        return {"status": snap["status"], "ready": rt.ready, "saturated_pools": snap["saturated_pools"], "pools": snap["pools"]}

    @app.get("/readyz")
    async def readyz(request: Request):
        rt = runtime(request)
        deps = await rt.dependencies()
        ok = rt.ready and all(deps.values())
        return JSONResponse({"ready": ok, "warmed_up": rt.ready, "dependencies": deps, "warm_seconds": rt.warm_seconds}, status_code=200 if ok else 503)

    return app


app = create_app()  # settings are read from the environment at startup, not at import

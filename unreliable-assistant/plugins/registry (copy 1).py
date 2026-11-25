"""Lifecycle plugin registry (docs/design.md Gate 0 'Lifecycle Plugins').

Plugins are pluggy plugins implementing any of three hooks:
    pre_execute(ctx)            stage PRE_EXECUTION   - may rewrite ctx.query or raise Block
    scan_retrieved(ctx, data)   stage PRE_EXECUTION   - sees retrieved chunks before the model does; returns data
    post_execute(ctx, result)   stage POST_EXECUTION  - may rewrite/replace the result or raise Block

pluggy itself only orders by tryfirst/trylast, so ordering here is by an explicit numeric `priority` (lower runs first)
taken from configuration (mcp_config.json), as are `fail_closed` and `enabled`.

fail_closed=True : an unexpected plugin error BLOCKS the request (the control could not be applied).
fail_closed=False: the error is recorded and the request continues (an observability/availability plugin).
Fail-closed plugins cannot be toggled at runtime at all: removing a security control needs a config change + restart.
Plugin errors are recorded by exception CLASS only, because exception text can echo user input (CLAUDE.md rule 4).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

import pluggy
from opentelemetry import trace

_tracer = trace.get_tracer("unreliable_assistant.plugins")
PROJECT = "unreliable_assistant"
hookspec = pluggy.HookspecMarker(PROJECT)
hookimpl = pluggy.HookimplMarker(PROJECT)
PLATFORM_TENANT = "platform-ops"  # audit tenant for platform-level events (plugin toggles)


class Stage(str, Enum):
    PRE_EXECUTION = "PRE_EXECUTION"
    POST_EXECUTION = "POST_EXECUTION"


class Block(Exception):
    """Raised by a plugin to stop the request. `public_message` is safe to show the user; it never contains input."""

    def __init__(self, code: str, public_message: str, category: str) -> None:
        super().__init__(code)
        self.code, self.public_message, self.category = code, public_message, category


class PluginConfigError(Exception):
    pass


class Hooks:
    @hookspec
    def pre_execute(self, ctx: "RequestContext") -> None: ...

    @hookspec
    def scan_retrieved(self, ctx: "RequestContext", data: dict) -> dict | None: ...

    @hookspec
    def post_execute(self, ctx: "RequestContext", result: Any) -> None: ...


@dataclass
class RequestContext:
    tenant_id: str
    user_id: str
    session_id: str
    query: str                     # raw on creation; the pii_redactor replaces it with the redacted text
    query_sha256: str = ""
    route: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)
    pii_counts: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)   # audit-worthy facts: counts/ids only, never text
    timings_ms: dict[str, float] = field(default_factory=dict)
    blocked: Block | None = None
    started: float = field(default_factory=time.perf_counter)  # request start; lets post-plugins measure true end-to-end time

    @classmethod
    def create(cls, tenant_id: str, user_id: str, session_id: str, query: str) -> "RequestContext":
        return cls(tenant_id, user_id, session_id, query, hashlib.sha256(query.encode()).hexdigest())

    def event(self, plugin: str, event_type: str, **payload: Any) -> None:
        self.events.append({"plugin": plugin, "type": event_type, "payload": payload})


@dataclass
class PluginMeta:
    name: str
    stages: frozenset[Stage]
    priority: int
    fail_closed: bool
    enabled: bool = True
    background: bool = False  # POST plugins only: run after the response is ready, off the critical path


class AuditSink:  # interface only: StateStoreAuditSink in agents/subagents.py satisfies it
    async def record(self, tenant_id: str, actor: str, event_type: str, payload: dict[str, Any]) -> None: ...


Runner = Callable[..., Awaitable[Any]]


class PluginRegistry:
    def __init__(self, audit_sink: AuditSink | None = None, runner: Runner | None = None) -> None:
        self.pm = pluggy.PluginManager(PROJECT)
        self.pm.add_hookspecs(Hooks)
        self.meta: dict[str, PluginMeta] = {}
        self.sink = audit_sink
        self._runner = runner  # e.g. pools.triage.run_sync: keeps CPU-bound plugins off the event loop
        self._lock = threading.Lock()
        self._pending: list[tuple[str, str, str, dict[str, Any]]] = []
        self._bg: set[asyncio.Task] = set()

    # ---- registration & configuration -------------------------------------------------------------------------------
    def register(self, plugin: Any, meta: PluginMeta) -> None:
        if meta.name in self.meta:
            raise PluginConfigError(f"plugin already registered: {meta.name}")
        self.pm.register(plugin, name=meta.name)
        self.meta[meta.name] = meta

    @classmethod
    def from_config(cls, path: str | Path, factories: dict[str, Callable[[], Any]], audit_sink: AuditSink | None = None,
                    runner: Runner | None = None) -> "PluginRegistry":
        cfg = json.loads(Path(path).read_text())
        reg = cls(audit_sink, runner)
        for entry in cfg["plugins"]:
            name = entry["name"]
            if name not in factories:
                raise PluginConfigError(f"no implementation for configured plugin '{name}'")
            meta = PluginMeta(name, frozenset(Stage(s) for s in entry["stages"]), int(entry["priority"]),
                              bool(entry["fail_closed"]), bool(entry.get("enabled", True)), bool(entry.get("background", False)))
            if meta.fail_closed and not meta.enabled:
                raise PluginConfigError(f"fail-closed plugin '{name}' may not be disabled in configuration")
            reg.register(factories[name](), meta)
        return reg

    def require(self, names: list[str]) -> None:
        """Startup check: every named plugin must be registered and enabled, or the service must not start."""
        missing = [n for n in names if n not in self.meta or not self.meta[n].enabled]
        if missing:
            raise PluginConfigError(f"required plugins missing or disabled: {missing}")

    def set_enabled(self, name: str, enabled: bool, actor: str) -> None:
        """Hot-toggle a non-critical plugin. Fail-closed (security) plugins refuse."""
        with self._lock:
            m = self.meta.get(name)
            if m is None:
                raise KeyError(name)
            if m.fail_closed:
                raise PermissionError(f"'{name}' is fail-closed and cannot be toggled at runtime")
            if m.enabled != enabled:
                m.enabled = enabled
                self._pending.append((PLATFORM_TENANT, f"operator:{actor}", "plugin.toggled",
                                      {"plugin": name, "enabled": enabled}))

    def status(self) -> list[dict[str, Any]]:
        return [{"name": m.name, "stages": sorted(s.value for s in m.stages), "priority": m.priority,
                 "fail_closed": m.fail_closed, "enabled": m.enabled} for m in sorted(self.meta.values(), key=lambda x: x.priority)]

    # ---- execution --------------------------------------------------------------------------------------------------
    def _impls(self, hook: str, stage: Stage) -> list[tuple[PluginMeta, Any]]:
        out = []
        for impl in getattr(self.pm.hook, hook).get_hookimpls():
            m = self.meta.get(impl.plugin_name)
            if m and m.enabled and stage in m.stages:
                out.append((m, impl))
        return sorted(out, key=lambda p: p[0].priority)

    async def _call(self, impl: Any, **kwargs: Any) -> Any:
        args = [kwargs[a] for a in impl.argnames]
        if self._runner is not None:
            return await self._runner(impl.function, *args)
        return await asyncio.to_thread(impl.function, *args)

    async def _run(self, hook: str, stage: Stage, ctx: RequestContext, background: bool | None = None, **kwargs: Any) -> Any:
        data = kwargs.get("data")
        for meta, impl in self._impls(hook, stage):
            if background is not None and meta.background != background:
                continue
            t0 = time.perf_counter()
            with _tracer.start_as_current_span(f"plugin.{meta.name}") as span:
                span.set_attribute("plugin.hook", hook)
                span.set_attribute("plugin.priority", meta.priority)
                try:
                    res = await self._call(impl, ctx=ctx, **kwargs)
                    if hook == "scan_retrieved" and res is not None:
                        kwargs["data"] = data = res
                    span.set_attribute("plugin.outcome", "ok")
                except Block as b:
                    ctx.blocked = b
                    span.set_attribute("plugin.outcome", "blocked")
                    span.set_attribute("plugin.block_code", b.code)
                    ctx.event(meta.name, "request.blocked", code=b.code, category=b.category)
                    ctx.timings_ms[meta.name] = (time.perf_counter() - t0) * 1000
                    break
                except Exception as e:  # noqa: BLE001
                    ctx.event(meta.name, "plugin.failed", error_type=type(e).__name__, fail_closed=meta.fail_closed)
                    span.set_attribute("plugin.outcome", "failed_closed" if meta.fail_closed else "failed_open")
                    if meta.fail_closed:
                        ctx.blocked = Block("E_PLUGIN_FAILED", "The request could not be processed safely.", "plugin_failure")
                        ctx.timings_ms[meta.name] = (time.perf_counter() - t0) * 1000
                        break
            ctx.timings_ms[meta.name] = (time.perf_counter() - t0) * 1000
        return data

    async def run_pre(self, ctx: RequestContext) -> RequestContext:
        with _tracer.start_as_current_span("hook.pre") as span:  # KPI-01 measures this span
            t0 = time.perf_counter()
            await self._run("pre_execute", Stage.PRE_EXECUTION, ctx)
            ctx.timings_ms["hook.pre"] = (time.perf_counter() - t0) * 1000
            span.set_attribute("hook.blocked", ctx.blocked is not None)
        await self._flush(ctx)
        return ctx

    async def run_retrieved(self, ctx: RequestContext, data: dict) -> dict:
        out = await self._run("scan_retrieved", Stage.PRE_EXECUTION, ctx, data=data)
        await self._flush(ctx)
        return out if out is not None else data

    async def run_post(self, ctx: RequestContext, result: Any) -> Any:
        with _tracer.start_as_current_span("hook.post"):
            await self._run("post_execute", Stage.POST_EXECUTION, ctx, background=False, result=result)
            if ctx.blocked is not None and getattr(result, "outcome", None) != "BLOCKED":
                result.outcome, result.answer = "BLOCKED", ctx.blocked.public_message  # post-stage veto replaces the output
        await self._flush(ctx)
        if any(m.background for m, _ in self._impls("post_execute", Stage.POST_EXECUTION)):
            task = asyncio.create_task(self._run("post_execute", Stage.POST_EXECUTION, ctx, background=True, result=result))
            self._bg.add(task)  # strong reference until done; the task inherits the request's trace context
            task.add_done_callback(self._bg.discard)
        return result

    async def drain(self) -> None:
        """Wait for background plugins (used at shutdown and in tests)."""
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)

    async def _flush(self, ctx: RequestContext) -> None:
        events, ctx.events = ctx.events, []
        batch = [(ctx.tenant_id, f"plugin:{e['plugin']}", e["type"], e["payload"]) for e in events]
        with self._lock:
            batch += self._pending
            self._pending = []
        if self.sink is None:
            return
        for tenant, actor, kind, payload in batch:
            try:
                await self.sink.record(tenant, actor, kind, payload)
            except Exception:  # auditing must not break serving; the failure itself is visible in traces
                with _tracer.start_as_current_span("plugin.audit_failed"):
                    pass

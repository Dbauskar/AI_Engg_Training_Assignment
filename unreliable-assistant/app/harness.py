"""ToolExecutionHarness: the only path by which a tool may be invoked (CLAUDE.md rule 2).

Pipeline (docs/design.md v1.1, Gate 4):
  0. tenant guard      -> E_TENANT_MISSING / E_TENANT_MISMATCH
  1. strict Pydantic v2 validation (first pass)
  2. deterministic regex auto-repair, then re-validate  -> E_SCHEMA_INVALID if still invalid
  3. RBAC tier check                                    -> E_PERMISSION_DENIED
  4. UUIDv4 idempotency (replay returns stored result)  -> E_IDEMPOTENT_REPLAY while in flight
Handlers only run after all four pass. Errors are structured, never raw exceptions, and never
contain raw input values (CLAUDE.md rule 4).
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import IntEnum
from typing import Any, Awaitable, Callable, Protocol

from opentelemetry import trace
from pydantic import BaseModel, ValidationError

from app.contracts import TENANT_RE, UUID4_RE

_tracer = trace.get_tracer("unreliable_assistant.harness")
_TENANT = re.compile(TENANT_RE)
_UUID4 = re.compile(UUID4_RE)

IDEMPOTENCY_TTL_S = 24 * 3600
_PENDING = "__PENDING__"


class Tier(IntEnum):
    TIER_1_READ = 1
    TIER_2_COMPUTE = 2
    TIER_3_MUTATE = 3


# --- structured errors (design Section 2.5) ---------------------------------
ERROR_TABLE: dict[str, tuple[int, bool, str]] = {
    # code: (json-rpc code, retryable, remediation)
    "E_SCHEMA_INVALID": (-32602, False, "Fix the listed fields and resend."),
    "E_TENANT_MISSING": (-32001, False, "Provide a valid tenant_id."),
    "E_TENANT_MISMATCH": (-32002, False, "tenant_id must match the authenticated caller."),
    "E_PERMISSION_DENIED": (-32003, False, "Caller tier is below the tool tier."),
    "E_PII_DETECTED": (-32007, False, "Remove or redact personal data before storing it."),
    "E_NOT_GROUNDED": (-32005, False, "Cite at least one valid retrieved chunk for this fact."),
    "E_RATE_LIMITED": (-32020, True, "System is at capacity; retry with backoff."),
    "E_IDEMPOTENT_REPLAY": (-32004, True, "Request is already in flight; retry shortly."),
    "E_UPSTREAM_UNAVAILABLE": (-32031, True, "Dependency unavailable or needs rebuilding; retry later."),
    "E_INTERNAL": (-32603, True, "Retry once; quote the trace ID if it persists."),
}


class ToolFailure(Exception):
    """Raised by tool handlers for expected failures; the harness maps `code` to a structured error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.public_message = code, message


class ToolError(BaseModel):
    code: str
    rpc_code: int
    message: str
    retryable: bool
    remediation: str
    fields: list[str] = []  # field names only, never values


def _err(code: str, message: str, fields: list[str] | None = None) -> ToolError:
    rpc, retry, fix = ERROR_TABLE[code]
    return ToolError(code=code, rpc_code=rpc, message=message, retryable=retry,
                     remediation=fix, fields=fields or [])


make_error = _err  # public alias for subagent wrappers


class HarnessResult(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    repaired: bool = False
    replayed: bool = False


# --- caller & tool registration ---------------------------------------------
@dataclass(frozen=True)
class Caller:
    tenant_id: str
    user_id: str
    max_tier: Tier


Handler = Callable[[Any, Caller], Awaitable[BaseModel | dict[str, Any]]]
Repairer = Callable[[Any], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    tier: Tier
    input_model: type[BaseModel]
    handler: Handler
    # field name -> repairer; defaults come from DEFAULT_REPAIRERS
    repairers: dict[str, Repairer] = field(default_factory=dict)


# --- deterministic auto-repair (design 2.5) ---------------------------------
_SKU = re.compile(r"^\s*sku[\s_-]*(\d{1,3})\s*$", re.I)
_LOT = re.compile(r"^\s*lot[\s_-]*(\d{1,5})\s*$", re.I)


def repair_sku(v: Any) -> Any:
    if isinstance(v, str) and (m := _SKU.match(v)):
        return f"SKU-{int(m.group(1)):03d}"
    return v


def repair_lot(v: Any) -> Any:
    if isinstance(v, str) and (m := _LOT.match(v)):
        return f"LOT-{int(m.group(1)):05d}"
    return v


def repair_currency(v: Any) -> Any:
    return v.strip().upper() if isinstance(v, str) else v


def repair_amount(v: Any) -> Any:
    """Normalise to a string decimal. Never rounds money: >2 decimals is left unrepaired."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        v = str(v)
    if not isinstance(v, str):
        return v
    s = v.strip().replace(",", "")
    try:
        d = Decimal(s)
        if not d.is_finite() or d < 0 or d != d.quantize(Decimal("0.01")):
            return v
        return format(d.normalize() if d == d.to_integral() else d.quantize(Decimal("0.01")), "f")  # "1250.5000" -> "1250.50"
    except InvalidOperation:  # e.g. 40 digits: quantize exceeds the context precision. Leave it for validation to reject.
        return v


DEFAULT_REPAIRERS: dict[str, Repairer] = {
    "sku": repair_sku,
    "lot": repair_lot,
    "currency": repair_currency,
    "amount": repair_amount,
}


def _repair(raw: dict[str, Any], spec: ToolSpec) -> dict[str, Any]:
    reps = {**DEFAULT_REPAIRERS, **spec.repairers}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            v = v.strip()
        if k in reps:
            v = reps[k](v)
        out[k] = v
    return out


# --- idempotency stores ------------------------------------------------------
class IdempotencyStore(Protocol):
    async def claim(self, key: str) -> bool: ...
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str) -> None: ...
    async def release(self, key: str) -> None: ...


class InMemoryIdempotencyStore:
    """Single-process store for tests and local runs."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    async def claim(self, key: str) -> bool:
        if key in self._d:
            return False
        self._d[key] = _PENDING
        return True

    async def get(self, key: str) -> str | None:
        return self._d.get(key)

    async def set(self, key: str, value: str) -> None:
        self._d[key] = value

    async def release(self, key: str) -> None:
        self._d.pop(key, None)


class RedisIdempotencyStore:
    """Redis-backed store. Keys use a {tenant} hash tag so a tenant's keys share a slot."""

    def __init__(self, client: Any, ttl_s: int = IDEMPOTENCY_TTL_S) -> None:
        self._r = client
        self._ttl = ttl_s

    async def claim(self, key: str) -> bool:
        return bool(await self._r.set(key, _PENDING, nx=True, ex=self._ttl))

    async def get(self, key: str) -> str | None:
        v = await self._r.get(key)
        return v.decode() if isinstance(v, bytes) else v

    async def set(self, key: str, value: str) -> None:
        await self._r.set(key, value, ex=self._ttl)

    async def release(self, key: str) -> None:
        await self._r.delete(key)


def idempotency_key(tenant_id: str, request_id: str) -> str:
    return f"idem:{{{tenant_id}}}:{request_id}"


# --- stats (feeds KPI-14 / KPI-15) -------------------------------------------
@dataclass
class HarnessStats:
    calls: int = 0
    first_pass_valid: int = 0
    repair_attempted: int = 0
    repair_succeeded: int = 0

    @property
    def first_pass_rate(self) -> float:
        return self.first_pass_valid / self.calls if self.calls else 0.0

    @property
    def repair_success_rate(self) -> float:
        return self.repair_succeeded / self.repair_attempted if self.repair_attempted else 0.0


def _field_names(e: ValidationError) -> list[str]:
    """Field names only: pydantic error text can echo the offending value, so never use it."""
    names: list[str] = []
    for err in e.errors(include_input=False, include_url=False, include_context=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        if loc not in names:
            names.append(loc)
    return names


# --- the harness -------------------------------------------------------------
class ToolExecutionHarness:
    def __init__(self, store: IdempotencyStore | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._store: IdempotencyStore = store or InMemoryIdempotencyStore()
        self.stats = HarnessStats()

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def input_fields(self, tool_name: str) -> frozenset[str] | None:
        """Field names a tool accepts (lets callers inject session_id etc. without touching private state)."""
        spec = self._tools.get(tool_name)
        return frozenset(spec.input_model.model_fields) if spec else None

    async def call(self, tool_name: str, raw_args: dict[str, Any], caller: Caller) -> HarnessResult:
        with _tracer.start_as_current_span("tool.harness") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("caller.tier", int(caller.max_tier))
            res = await self._run(tool_name, raw_args, caller, span)
            span.set_attribute("harness.ok", res.ok)
            span.set_attribute("harness.repaired", res.repaired)
            span.set_attribute("harness.replayed", res.replayed)
            if res.error:
                span.set_attribute("harness.error_code", res.error.code)
            return res

    async def _run(self, tool_name: str, raw: dict[str, Any], caller: Caller, span: Any) -> HarnessResult:
        spec = self._tools.get(tool_name)
        if spec is None:
            return HarnessResult(ok=False, error=_err("E_SCHEMA_INVALID", "Unknown tool.", ["tool"]))
        span.set_attribute("tool.tier", int(spec.tier))

        # 0. tenant guard (rule 3)
        tid = raw.get("tenant_id") if isinstance(raw, dict) else None
        if not isinstance(tid, str) or not _TENANT.match(tid) or not _TENANT.match(caller.tenant_id):
            return HarnessResult(ok=False, error=_err("E_TENANT_MISSING", "tenant_id missing or malformed."))
        if tid != caller.tenant_id:
            return HarnessResult(ok=False, error=_err("E_TENANT_MISMATCH", "tenant_id does not match caller."))
        span.set_attribute("tenant.id", tid)

        # 1. strict validation, first pass
        self.stats.calls += 1
        repaired = False
        try:
            model = spec.input_model.model_validate(raw)
            self.stats.first_pass_valid += 1
        except ValidationError:
            # 2. deterministic auto-repair, then re-validate
            self.stats.repair_attempted += 1
            try:
                model = spec.input_model.model_validate(_repair(raw, spec))
                repaired = True
                self.stats.repair_succeeded += 1
            except ValidationError as e2:
                return HarnessResult(
                    ok=False,
                    error=_err("E_SCHEMA_INVALID", "Arguments failed validation.", _field_names(e2)),
                )

        # 3. RBAC tier
        if caller.max_tier < spec.tier:
            return HarnessResult(ok=False, repaired=repaired, error=_err(
                "E_PERMISSION_DENIED", f"Tool requires {spec.tier.name}."))

        # 4. idempotency
        rid = getattr(model, "request_id", None)
        if not isinstance(rid, str) or not _UUID4.match(rid):
            return HarnessResult(ok=False, repaired=repaired, error=_err(
                "E_SCHEMA_INVALID", "request_id must be a UUIDv4.", ["request_id"]))
        key = idempotency_key(caller.tenant_id, rid)
        try:
            claimed = await self._store.claim(key)
        except asyncio.CancelledError:
            raise
        except Exception:  # idempotency store down (FMEA #1): fail closed, a mutation must never run unprotected
            span.set_attribute("harness.idempotency_store_down", True)
            return HarnessResult(ok=False, repaired=repaired, error=_err(
                "E_UPSTREAM_UNAVAILABLE", "Idempotency store unavailable; request not executed."))
        if not claimed:
            stored = await self._store.get(key)
            if stored and stored != _PENDING:
                return HarnessResult(ok=True, data=json.loads(stored), repaired=repaired, replayed=True)
            return HarnessResult(ok=False, repaired=repaired, error=_err(
                "E_IDEMPOTENT_REPLAY", "Request with this request_id is already in flight."))

        try:
            out = await spec.handler(model, caller)
        except asyncio.CancelledError:  # caller timed out/cancelled: do not leave the key stuck as 'in flight'
            await self._store.release(key)
            raise
        except ToolFailure as tf:
            await self._store.release(key)
            return HarnessResult(ok=False, repaired=repaired, error=_err(tf.code, tf.public_message))
        except Exception:  # never leak raw exceptions or input echoes
            await self._store.release(key)
            span.set_attribute("harness.handler_exception", True)
            return HarnessResult(ok=False, repaired=repaired, error=_err("E_INTERNAL", "Tool handler failed."))
        data = out.model_dump(mode="json") if isinstance(out, BaseModel) else out
        try:
            await self._store.set(key, json.dumps(data, sort_keys=True))
        except Exception:  # result already produced; the key expires from PENDING on its TTL, so a retry cannot double-execute
            span.set_attribute("harness.idempotency_store_set_failed", True)
        return HarnessResult(ok=True, data=data, repaired=repaired)

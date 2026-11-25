STATUS: LOCKED

# Design Document: The Unreliable Assistant → Enterprise Grade

| Item | Detail |
|---|---|
| Version | 1.8 |
| Date | 2026-11-18 |
| Source | Mid-Check Enterprise Blueprint v2; Post-Hardening Audit Report |
| Scope | Gate 1 hard-gate artifact. No implementation code may be written until this file is LOCKED (it is). |
| Change policy | Any change after locking requires a new ADR (Section 7), a version bump, and a note in the change log at the end of this file. |

Targets: end-to-end p95 ≤ 450 ms and p99 ≤ 600 ms at 50,000 concurrent virtual users (VUs); 36 stage-level KPIs; zero cross-tenant leakage; zero autonomous state-changing action.

---

## 1. System Charter, Non-Goals, Personas

### 1.1 Charter
Provide grounded, cited answers over an enterprise policy and product knowledge base, and allow a *human-approved* set of workflow actions, through a hybrid retrieval pipeline, a bounded agent loop, and deterministic governance controls. When evidence is insufficient the system abstains instead of guessing.

### 1.2 Hard boundaries (invariants)
| ID | Invariant |
|---|---|
| B1 | **No autonomous financial, clinical, or legal execution.** Every `TIER_3_MUTATE` action is frozen as `PENDING_APPROVAL` until an authorized human approves. |
| B2 | Every tool call passes through `ToolExecutionHarness` (validate → auto-repair → RBAC tier → idempotency). |
| B3 | `tenant_id` is mandatory in every DB query, cache key, audit record and trace attribute. A call without it is rejected (`E_TENANT_MISSING`). |
| B4 | Raw PII is never logged, traced, cached, or persisted. Redaction happens before any of those. |
| B5 | The ReAct loop is bounded (5 iterations, 450 ms total). No unbounded recursion. |
| B6 | If the top reranker score is < 0.42 the system abstains deterministically, without LLM generation. |
| B7 | Retrieval is read-only. The Retrieval subagent can never invoke a mutating tool. |
| B8 | Audit log is append-only and SHA-256 hash-chained. |

### 1.3 Non-goals
- Autonomous trading, payments, diagnosis, prescribing, or legal advice.
- Free-form web browsing or arbitrary code execution by the agent.
- Training or fine-tuning models (LoRA adapters are a roadmap item, not in scope).
- Multi-region replication and TensorRT-LLM serving (roadmap; not required for the KPI gate).
- Cross-tenant analytics or any shared per-tenant cache.
- Serving users with no `tenant_id`.

### 1.4 Personas
| Persona | Role / permission tier | Goals | Constraints |
|---|---|---|---|
| End User (Employee/Customer) | `TIER_1_READ` | Ask policy/product questions, get cited answers | Cannot trigger mutations directly |
| Operator / Analyst | `TIER_1_READ`, `TIER_2_COMPUTE` | Run calculations, validate forms, draft requests | Compute only; no state change |
| Requester | Can *submit* Tier 3 requests | Initiate a transaction | Request is frozen until approved; cannot self-approve |
| Approver (Compliance Officer) | HITL reviewer | Approve/reject `PENDING_APPROVAL` items | Cannot approve own requests (separation of duties) |
| Platform/SRE Engineer | Observability, runbooks | Keep SLOs, respond to incidents | No access to raw PII or tenant data content |
| Security/Red-Team Engineer | Test harness | Run adversarial and chaos suites | Only against non-production or own local instance |
| Auditor | Read-only audit | Verify hash chain, KPI evidence | Read-only |

---

## 2. MCP Tool Contracts

Transport: MCP over JSON-RPC 2.0. `hybrid-rag` and `graphify-vault` use stdio; `state-store` uses streamable HTTP. All requests carry `tenant_id`, `request_id` (UUIDv4) and W3C `traceparent`. Tool inputs use strict Pydantic v2 models (`extra="forbid"`, `strict=True` unless noted).

### 2.1 Shared types and regex constraints
| Name | Pattern | Notes |
|---|---|---|
| `tenant_id` | `^[a-z0-9][a-z0-9_-]{2,39}$` | Lowercase slug |
| `user_id` | `^[A-Za-z0-9_.-]{1,64}$` | Opaque ID, never an email |
| `session_id` | `^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$` | UUIDv4 |
| `request_id` / idempotency key | same as `session_id` | UUIDv4, required on every call |
| `chunk_id` | `^[a-z0-9]{2,16}-[0-9]{4}-[0-9a-f]{8}$` | `docid-seq-hash8` |
| `fact_key` | `^[a-z][a-z0-9_.]{2,63}$` | Dotted lower-case key |
| `sku` | `^SKU-\d{3}$` | Auto-repair: `SKU99` → `SKU-099` |
| `lot` | `^LOT-\d{5}$` | Auto-repair: `lot 44` → `LOT-00044` |
| `account_id` | `^ACC-\d{8}$` | |
| `currency` | `^[A-Z]{3}$` | ISO 4217 |
| `amount` | decimal string `^\d{1,12}(\.\d{1,2})?$` | Never float |

### 2.2 Pydantic v2 models (specification)

```python
# Specification only — implemented later under app/ and mcp_servers/
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Tenant  = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{2,39}$")]
UserId  = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
UUID4   = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")]
FactKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.]{2,63}$")]

class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    tenant_id: Tenant
    user_id: UserId
    request_id: UUID4

class HybridRetrieveIn(_Base):
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    top_k_sparse: int = Field(50, ge=1, le=100)
    top_k_dense: int = Field(50, ge=1, le=100)
    rrf_k: int = Field(60, ge=1, le=200)
    rerank_k: int = Field(4, ge=1, le=10)
    score_threshold: float = Field(0.42, ge=0.0, le=1.0)
    domain: str | None = Field(None, pattern=r"^[a-z][a-z0-9_]{1,31}$")

class RetrievedChunk(BaseModel):
    chunk_id: str = Field(pattern=r"^[a-z0-9]{2,16}-[0-9]{4}-[0-9a-f]{8}$")
    text: str
    score: float = Field(ge=0, le=1)          # sigmoid cross-encoder score
    bm25_rank: int | None
    dense_rank: int | None
    source_uri: str

class HybridRetrieveOut(BaseModel):
    abstain: bool                              # true if top score < threshold
    chunks: list[RetrievedChunk]               # <= rerank_k
    top_score: float
    stage_latency_ms: dict[Literal["bm25","dense","rrf","rerank"], float]

class GetSessionFactsIn(_Base):
    session_id: UUID4
    keys: list[FactKey] | None = Field(None, max_length=50)   # None = all verified facts
    limit: int = Field(50, ge=1, le=200)

class SessionFact(BaseModel):
    fact_key: str
    value: str                                  # already PII-redacted
    verified: bool
    created_at: str                             # RFC 3339

class GetSessionFactsOut(BaseModel):
    facts: list[SessionFact]

class WriteSessionFactIn(_Base):
    session_id: UUID4
    fact_key: FactKey
    value: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    source_chunk_ids: list[str] = Field(min_length=1, max_length=8)   # must be grounded
    verified: Literal[True]                    # only verified facts may be persisted

class WriteSessionFactOut(BaseModel):
    stored: bool
    fact_key: str
    audit_hash: str                             # chain head after write

class SubmitTransactionIn(_Base):
    session_id: UUID4
    action: Literal["TRANSFER", "ORDER", "STATUS_CHANGE"]
    account_id: str = Field(pattern=r"^ACC-\d{8}$")
    amount: str | None = Field(None, pattern=r"^\d{1,12}(\.\d{1,2})?$")
    currency: str | None = Field(None, pattern=r"^[A-Z]{3}$")
    justification: Annotated[str, StringConstraints(min_length=10, max_length=500)]

class SubmitTransactionOut(BaseModel):
    status: Literal["PENDING_APPROVAL"]        # the ONLY status this tool may return
    approval_id: UUID4
    audit_hash: str
```

### 2.3 JSON schemas (input; generated from the models above and checked into `docs/schemas/` at build time)

**`hybrid_retrieve`** (tier `TIER_1_READ`)
```json
{
  "type": "object", "additionalProperties": false,
  "required": ["tenant_id", "user_id", "request_id", "query"],
  "properties": {
    "tenant_id":  {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{2,39}$"},
    "user_id":    {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,64}$"},
    "request_id": {"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
    "query":      {"type": "string", "minLength": 1, "maxLength": 2000},
    "top_k_sparse": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
    "top_k_dense":  {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
    "rrf_k":        {"type": "integer", "minimum": 1, "maximum": 200, "default": 60},
    "rerank_k":     {"type": "integer", "minimum": 1, "maximum": 10,  "default": 4},
    "score_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.42},
    "domain": {"type": ["string", "null"], "pattern": "^[a-z][a-z0-9_]{1,31}$"}
  }
}
```

**`get_session_facts`** (tier `TIER_1_READ`)
```json
{
  "type": "object", "additionalProperties": false,
  "required": ["tenant_id", "user_id", "request_id", "session_id"],
  "properties": {
    "tenant_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{2,39}$"},
    "user_id": {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,64}$"},
    "request_id": {"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
    "session_id": {"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
    "keys": {"type": ["array", "null"], "maxItems": 50, "items": {"type": "string", "pattern": "^[a-z][a-z0-9_.]{2,63}$"}},
    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}
  }
}
```

**`write_session_fact`** (tier `TIER_2_COMPUTE`; writes only verified, grounded facts to the session scratch store, not business state)
```json
{
  "type": "object", "additionalProperties": false,
  "required": ["tenant_id", "user_id", "request_id", "session_id", "fact_key", "value", "source_chunk_ids", "verified"],
  "properties": {
    "tenant_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{2,39}$"},
    "user_id": {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,64}$"},
    "request_id": {"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
    "session_id": {"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
    "fact_key": {"type": "string", "pattern": "^[a-z][a-z0-9_.]{2,63}$"},
    "value": {"type": "string", "minLength": 1, "maxLength": 1000},
    "source_chunk_ids": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "string", "pattern": "^[a-z0-9]{2,16}-[0-9]{4}-[0-9a-f]{8}$"}},
    "verified": {"const": true}
  }
}
```

**`submit_transaction`** (tier `TIER_3_MUTATE`; always creates a `PENDING_APPROVAL` record and never executes)
```json
{
  "type": "object", "additionalProperties": false,
  "required": ["tenant_id", "user_id", "request_id", "session_id", "action", "account_id", "justification"],
  "properties": {
    "tenant_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{2,39}$"},
    "user_id": {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,64}$"},
    "request_id": {"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
    "session_id": {"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
    "action": {"enum": ["TRANSFER", "ORDER", "STATUS_CHANGE"]},
    "account_id": {"type": "string", "pattern": "^ACC-\\d{8}$"},
    "amount": {"type": ["string", "null"], "pattern": "^\\d{1,12}(\\.\\d{1,2})?$"},
    "currency": {"type": ["string", "null"], "pattern": "^[A-Z]{3}$"},
    "justification": {"type": "string", "minLength": 10, "maxLength": 500}
  }
}
```

### 2.4 Tool summary
| Tool | Server | Tier | Idempotent | Mutates business state | Allowed callers |
|---|---|---|---|---|---|
| `hybrid_retrieve` | `hybrid-rag` | 1 | Yes (read) | No | Retrieval subagent |
| `get_session_facts` | `state-store` | 1 | Yes (read) | No | Retrieval, Execution, Verification (audit-only) |
| `write_session_fact` | `state-store` | 2 | Yes (key = `request_id`) | Session scratch only | Execution, Verification |
| `submit_transaction` | `state-store` | 3 | Yes (key = `request_id`) | Creates pending record; **execution requires human approval** | Execution subagent only |

### 2.5 Error codes
Errors are returned as structured JSON-RPC errors `{code, message, data:{error_code, retryable, remediation}}`. Messages never include raw input values.

| Error code | JSON-RPC code | Meaning | Retryable | Fallback / remediation |
|---|---|---|---|---|
| `E_SCHEMA_INVALID` | -32602 | Pydantic validation failed and auto-repair could not fix it | No | Return field-level hint; counts toward circuit breaker |
| `E_TENANT_MISSING` | -32001 | `tenant_id` absent or malformed | No | Reject; security log |
| `E_TENANT_MISMATCH` | -32002 | Resource belongs to another tenant | No | Reject as not-found; security alert |
| `E_PERMISSION_DENIED` | -32003 | Caller tier below tool tier | No | Reject; audit |
| `E_IDEMPOTENT_REPLAY` | -32004 | `request_id` seen before | No | Return the stored result |
| `E_NOT_GROUNDED` | -32005 | `write_session_fact` without valid `source_chunk_ids` | No | Reject write |
| `E_APPROVAL_REQUIRED` | -32006 | Tier 3 action frozen as `PENDING_APPROVAL` | No | Normal outcome; surface approval ID |
| `E_ABSTAIN` | -32010 | Reranker top score below threshold | No | Deterministic abstention response |
| `E_LOOP_BREAKER` | -32011 | Same (tool, args-hash) failed twice | No | `HALTED_BREAKER` with remediation prompt |
| `E_RATE_LIMITED` | -32020 | Tenant or global rate limit exceeded | Yes | Backoff with jitter |
| `E_UPSTREAM_TIMEOUT` | -32030 | MCP call exceeded 150 ms | Yes (once) | Fallback per FMEA (Section 6) |
| `E_UPSTREAM_UNAVAILABLE` | -32031 | MCP server/DB/Redis down | Yes | Fallback per FMEA (Section 6) |
| `E_INTERNAL` | -32603 | Unexpected server fault | Yes (once) | Generic message; trace ID returned |

**Auto-repair rules (deterministic, applied only in the harness before validation fails):** `SKU` with 1–3 digits → zero-pad to 3 digits with `SKU-`; `lot <n>` → `LOT-` + 5-digit zero-pad; uppercase currency; strip whitespace; amount normalized to string decimal. Repair is logged as `repaired=true`. Anything else fails with `E_SCHEMA_INVALID`.

---

## 3. Topology and Data Flow

### 3.1 Components
| Component | Technology | Role | State |
|---|---|---|---|
| API Gateway | Envoy/NGINX | TLS, rate limit, routing to stateless workers | None |
| Agent API | FastAPI + Uvicorn (async) | Orchestrates the request, hosts MCP client | Stateless |
| Pre-Execution Plugins | pluggy | `pii_redactor` (p10) → `injection_shield` (p20) → `circuit_breaker` | None |
| Semantic Cache | Redis Cluster | Tenant-scoped response cache | Ephemeral |
| Triage Agent | Distilled classifier, temp 0 | Route: `DIRECT_ANSWER`, `HYBRID_RAG`, `TRANSACTION_EXEC`, `REJECT_OOS` | None |
| Retrieval Subagent | MCP client | Read-only; calls `hybrid-rag` and `graphify-vault` | None |
| Execution Subagent | MCP client | Validates and submits via `state-store`; Tier 3 freezes | None |
| Verification Subagent | Independent context | Claim-to-chunk NLI check; can veto drafts | None |
| `mcp-hybrid-rag` | MCP server (stdio) | BM25 + dense → RRF → ONNX cross-encoder | Read-only index |
| `mcp-graphify-vault` | MCP server (stdio) | 2-hop subgraph from Obsidian vault + code graph | Read-only graph |
| `mcp-state-store` | MCP server (streamable HTTP) | Facts, audit log, approvals | PostgreSQL (pgvector) |
| ONNX Rerank/Embed workers | ONNX Runtime FP16/INT8 | Batched inference (batch 32, max wait 8 ms) | Stateless |
| LLM | vLLM / hosted API | Generation (continuous batching, prefix cache) | Stateless |
| Post-Execution Plugins | pluggy | RAGAS, OTel export, audit append (async, off critical path) | None |
| Observability | OTel Collector → Prometheus + Tempo + Grafana | Traces, metrics, dashboards | Time series |

### 3.2 Request flow (happy path, `HYBRID_RAG`)
```
Client
  → API Gateway (TLS, rate limit)
  → Agent API  [traceparent started]
  → PRE: pii_redactor → injection_shield → circuit_breaker state check      (hook.pre)
  → Redis semantic cache lookup  key=SHA256(tenant:user:norm_query)         (cache.lookup)
        hit  → respond (skip rest)
        miss → continue
  → Triage Agent (temp 0)                                                    (triage.classify)
        DIRECT_ANSWER → LLM → respond
        REJECT_OOS    → fixed refusal → respond
        TRANSACTION_EXEC → Execution subagent → Tier 3 freeze (see 3.3)
        HYBRID_RAG    → continue
  → Retrieval subagent → mcp-hybrid-rag.hybrid_retrieve
        BM25 ‖ Dense (parallel) → RRF → Cross-Encoder rerank                 (mcp.rag.*)
        top score < 0.42 → abstain (deterministic, no LLM)
  → mcp-graphify-vault: ≤ 2-hop subgraph for context compression
  → Bounded ReAct loop (≤ 5 iterations, 450 ms total)                        (agent.react.step)
        GoalAnchorBlock at index 0; observations ≤ 180 tokens; last 3 turns hot
  → LLM generate (stream)                                                    (llm.generate)
  → Verification subagent: claim-to-chunk NLI; veto → re-answer extractively or abstain
  → Respond to client (stream)
  → ASYNC POST: RAGAS scoring, OTel export, audit-log append, cache write   (hook.post.*)
```

### 3.3 Tier 3 (transaction) flow
```
User request → PRE plugins → Triage=TRANSACTION_EXEC → Execution subagent
  → harness: validate + repair + RBAC(Tier 3) + idempotency
  → state-store.submit_transaction → INSERT approvals(status=PENDING_APPROVAL)
  → loop FROZEN, user told "awaiting approval, ID <approval_id>"
Approver (UI/API, different identity) → APPROVE / REJECT
  → approved: separate, audited executor performs the action
  → rejected: reason fed back to agent as an Observation; user informed
```
The agent never holds a code path that executes a Tier 3 action.

### 3.4 Data stores and isolation
| Store | Contents | Isolation rule |
|---|---|---|
| Redis | Cache entries, idempotency keys, breaker counters, rate counters | Key prefix includes `tenant_id`; hash includes `tenant_id:user_id` |
| PostgreSQL `session_facts` | Verified, redacted facts | `tenant_id` column in every query; row-level security enabled |
| PostgreSQL `audit_log` | Hash-chained events | Append-only; `tenant_id` + `prev_hash` + `hash` |
| PostgreSQL `approvals` | Pending/decided Tier 3 requests | `tenant_id`; requester ≠ approver constraint |
| Vector/BM25 indexes | Chunked policy documents | Per-tenant namespace filter at query time |

---

## 4. End-to-End Latency Budget

Goal: **p95 ≤ 450 ms, p99 ≤ 600 ms** at 50,000 concurrent VUs for the `HYBRID_RAG` path (the slowest common path).

### 4.1 Critical-path budget (cache miss, `HYBRID_RAG`)
BM25 and dense retrieval run in parallel, so the critical path takes the larger of the two.

| # | Stage | Budget (p95, ms) | On critical path? | Notes |
|---|---|---|---|---|
| 1 | Pre-hooks (PII + injection) | 15 | Yes | KPI-01 |
| 2 | Semantic cache lookup | 5 | Yes | Redis, same AZ |
| 3 | Triage classification | 25 | Yes | KPI-05 |
| 4 | BM25 (stage 1) | 12 | Parallel with 5 | KPI-07 |
| 5 | Dense bi-encoder + FAISS (stage 2) | 25 | Yes (max of 4, 5) | KPI-08; counts 25 |
| 6 | RRF fusion | 3 | Yes | |
| 7 | Cross-encoder rerank (stage 3) | 45 | Yes | KPI-09 |
| 8 | MCP JSON-RPC overhead | 4 | Yes | KPI-30; one retrieve call on the path |
| 9 | Graphify subgraph lookup | 10 | Yes | Overlapped with rerank when possible; budgeted conservatively |
| 10 | LLM generation to last token | 250 | Yes | Remainder, see 4.2 |
| 11 | Gateway, network, serialization, verification (NLI) | 68 | Yes | Reserve |
| | **Sum (critical path)** | **450** | | |

Arithmetic: 15 + 5 + 25 + 25 + 3 + 45 + 4 + 10 + 250 + 68 = **450 ms**. Retrieval on the critical path (rows 5–8: dense 25 + RRF 3 + rerank 45 + MCP 4) is 77 ms, because BM25 (row 4) overlaps with dense. Everything before the LLM (rows 1–9) is 132 ms.

### 4.2 LLM remainder
LLM remainder = 450 − (15+5+25+25+3+45+4+10) − reserve 68 = **250 ms**, which must cover generation to the last token for a ≤ ~120-token answer with a ~320-token Graphify context. TTFT target is 180 ms (KPI-25), leaving ~70 ms for streaming the rest. Answers needing longer generation are streamed, and the SLO is measured to the *last token* for the load test, which is the stricter reading.

### 4.3 Tail (p99) budget
| Component | p95 | p99 allowance | Reason |
|---|---|---|---|
| Retrieval + hooks + triage (rows 1–9) | 132 | 150 | Queueing under load; batching wait ≤ 8 ms already included |
| LLM generation | 250 | 330 | Continuous-batching tail |
| Reserve | 68 | 120 | GC, network jitter, NLI |
| **Total** | **450** | **600** | KPI-33 target ≤ 590 ms leaves 10 ms headroom |

### 4.4 Alternative paths
| Path | Target p95 | Mechanism |
|---|---|---|
| Cache hit | ≤ 60 ms | Hooks 15 + cache 5 + response, no LLM |
| Abstention | ≤ 90 ms | Retrieval completes, score < 0.42, deterministic response, no LLM |
| `REJECT_OOS` | ≤ 60 ms | Hooks + triage + fixed refusal |
| `TRANSACTION_EXEC` | ≤ 300 ms | Hooks + triage + harness + DB insert; no RAG |

### 4.5 Loop limits
`max_iterations = 5`, `per_step_timeout = 120 ms`, `total_loop_budget = 450 ms`. The typical task takes ≤ 2.2 steps (KPI-17); the budget in 4.1 assumes a single retrieve + single generate.

### 4.6 Honest caveats
- Summing per-stage p95s overstates the true p95 of the total (conservative), but queueing at 50K VUs can break per-stage assumptions. **This budget is a design target and is not proven until KPI-31..33 are measured under load.**
- The 250 ms LLM allowance assumes a fast served model with continuous batching; an unbatched large hosted model will not meet it. The model choice is validated by the load test, and the fallback is a smaller model or shorter answers (ADR-3 follow-up).
- The blueprint lists the triage budget as both 25 ms and 30 ms. This design uses **25 ms** (the stricter KPI-05).
- The blueprint says "3-stage" and "4-stage" retrieval for the same pipeline. This design defines it as **3 retrieval stages (BM25 ‖ dense → RRF → rerank)**, where BM25 and dense are two lanes of stage 1.

---

## 5. KPI and SLA Gate Matrix (36 KPIs)

Every KPI must pass in CI/CD (where measurable) and appear in the Streamlit scorecard (Tab 6). "Full scale" KPIs (31–36) require the 50K-VU distributed run; locally they are run at reduced scale and labelled as such.

| KPI | Stage | Metric | Threshold | Measurement tool |
|---|---|---|---|---|
| KPI-01 | 1 Ingress & Triage | Pre-hook guardrail latency (p95) | ≤ 15 ms | OTel span `hook.pre` |
| KPI-02 | 1 | PII redaction recall (email/phone/SSN/obfuscated) | ≥ 99.8% | Presidio + regex audit |
| KPI-03 | 1 | Prompt-injection block rate (direct + indirect) | ≥ 99.5% | Red-team benchmark |
| KPI-04 | 1 | Triage routing accuracy | ≥ 97.0% | Labeled intent set |
| KPI-05 | 1 | Triage classification latency (p95) | ≤ 25 ms | OTel span `triage.classify` |
| KPI-06 | 1 | False-positive refusal rate on benign queries | ≤ 1.0% | Benign control set |
| KPI-07 | 2 Hybrid Retrieval | Stage 1 BM25 latency (p95) | ≤ 12 ms | OTel span `mcp.rag.bm25` |
| KPI-08 | 2 | Stage 2 bi-encoder + FAISS latency (p95) | ≤ 25 ms | OTel span `mcp.rag.biencoder` |
| KPI-09 | 2 | Stage 3 cross-encoder rerank latency (p95) | ≤ 45 ms | OTel span `mcp.rag.rerank` |
| KPI-10 | 2 | Hybrid context recall@50 | ≥ 0.92 | RAGAS / IR eval |
| KPI-11 | 2 | Post-rerank context precision@4 | ≥ 0.88 | RAGAS evaluator |
| KPI-12 | 2 | Mean reciprocal rank (MRR@4) | ≥ 0.85 | IR benchmark script |
| KPI-13 | 2 | nDCG@4 | ≥ 0.86 | IR benchmark script |
| KPI-14 | 3 Agent & Loop | Tool schema first-pass validation rate | ≥ 96.0% | Pydantic telemetry |
| KPI-15 | 3 | Auto-repair success rate on malformed args | ≥ 90.0% | Harness metrics |
| KPI-16 | 3 | Infinite-loop escape rate (circuit breaker) | 100.0% | Chaos loop suite |
| KPI-17 | 3 | Mean ReAct steps per completed task | ≤ 2.2 | OTel trace count |
| KPI-18 | 3 | Context drift score (goal cosine alignment) | ≥ 0.85 | Embedding drift audit |
| KPI-19 | 3 | Tier 3 HITL gate enforcement rate | 100.0% | Audit-log integrity check |
| KPI-20 | 4 Generation & RAGAS | RAGAS faithfulness | ≥ 0.92 | Post-hook RAGAS |
| KPI-21 | 4 | RAGAS answer relevancy | ≥ 0.90 | Post-hook RAGAS |
| KPI-22 | 4 | RAGAS context entity recall | ≥ 0.85 | Post-hook RAGAS |
| KPI-23 | 4 | Unanswerable-query abstention accuracy | ≥ 98.0% | Negative test set |
| KPI-24 | 4 | Unsupported-claim / hallucination rate | ≤ 0.5% | NLI claim verifier |
| KPI-25 | 4 | Time-to-first-token (p95) | ≤ 180 ms | FastAPI stream metric |
| KPI-26 | 5 Token & Graphify | Prompt-token reduction via Graphify + Obsidian | ≥ 45.0% | Token-counter hook |
| KPI-27 | 5 | Semantic cache hit ratio (steady state) | ≥ 35.0% | Redis hit/miss metric |
| KPI-28 | 5 | Cross-tenant cache isolation integrity | 100.0% (0 leaks) | Concurrency probe |
| KPI-29 | 5 | Cost per 1,000 enterprise queries | ≤ $0.45 | OTel token-cost calc |
| KPI-30 | 5 | MCP JSON-RPC serialization overhead (p95) | ≤ 4 ms | MCP client timer |
| KPI-31 | 6 Scale & Load (50K) | E2E latency p50 @ 50K VUs | ≤ 240 ms | k6 + JMeter summary |
| KPI-32 | 6 | E2E latency p95 @ 50K VUs | ≤ 450 ms | k6 + JMeter summary |
| KPI-33 | 6 | E2E latency p99 @ 50K VUs | ≤ 590 ms (< 600) | k6 + JMeter summary |
| KPI-34 | 6 | HTTP 5xx error rate @ 50K VUs | ≤ 0.05% | k6 / JMeter threshold |
| KPI-35 | 6 | Sustained throughput | ≥ 8,500 RPS | k6 / JMeter counter |
| KPI-36 | 6 | Memory leak / RSS growth over 1-hr soak | ≤ 3.0% delta | Prometheus node exporter |

### 5.1 Release gates (RAGAS golden set, 273 questions)
Strata: Factoid 63 (23%), Complex Reasoning 62 (23%), Multi-Hop 50 (18%), Summary 38 (14%), Negative/Unanswerable 60 (22%). (v1.9: negatives raised from 37 to 60 because 21 test negatives made the abstention gate hinge on single items.) Gates: faithfulness ≥ 0.92, answer relevancy ≥ 0.90, context precision ≥ 0.88, context recall ≥ 0.85, context entity recall ≥ 0.85. A single missed gate fails the release.

### 5.2 Scale labelling rule
Each KPI result records `measured_at_scale` (e.g., `500 VUs`, `50,000 VUs`). The scorecard shows a KPI as **PASS (reduced scale)** rather than PASS if it was not measured at the target scale. KPI-31..36 can only be PASS at full scale.

---

## 6. Failure Mode and Effects Analysis (FMEA)

Severity (S), Occurrence (O), Detection (D) are rated 1–5 (5 = worst). Risk priority = S×O×D.

| # | Component / failure | Effect | S | O | D | RPN | Detection | Deterministic fallback |
|---|---|---|---|---|---|---|---|---|
| 1 | **Redis down** (cache, idempotency, breaker counters) | No cache; idempotency/breaker state lost | 3 | 2 | 1 | 6 | Health check, `E_UPSTREAM_UNAVAILABLE` | Bypass cache and go direct to RAG; idempotency falls back to Postgres unique constraint on `request_id`; breaker uses in-process counters; tighten rate limit; alert |
| 2 | **Reranker timeout / ONNX worker down** | No precision stage | 4 | 3 | 1 | 12 | Span > 45 ms ×3, timeout | Use RRF top-4 with abstain threshold raised to RRF-score floor; mark response `degraded=true`; extractive-quote-only answers |
| 3 | **MCP server crash/hang** (`hybrid-rag`) | No retrieval | 5 | 2 | 1 | 10 | JSON-RPC timeout > 150 ms, >1% rate | Trip bulkhead; fail over to cached read-replica index; restart worker; if none, abstain with "service degraded" |
| 4 | **LLM 429 / provider outage** | No generation | 4 | 3 | 1 | 12 | HTTP 429/5xx | Retry once with jittered backoff; fall back to secondary model; else return top reranked chunks as extractive quotes with citations |
| 5 | **PostgreSQL failover / unavailable** | No facts, audit, approvals | 5 | 2 | 1 | 10 | Pool errors, replica lag | Read-only mode from replica; **block all Tier 3 submissions (fail closed)**; queue audit events to a local WAL file and replay in order to preserve the hash chain |
| 6 | **Cache poisoning** (bad/injected answer stored) | Wrong answer served repeatedly | 5 | 2 | 3 | 30 | Verification-vetoed answers never cached; hash mismatch audit | Cache only responses that passed Verification and had faithfulness ≥ threshold; per-entry TTL; purge-by-tenant API; entries store source chunk hashes and are invalidated when sources change |
| 7 | **Cross-tenant key collision / leak** | Data exposure | 5 | 1 | 2 | 10 | Concurrency probe (KPI-28), key-prefix assertion | Key includes `SHA256(tenant:user:query)` plus a tenant prefix; any response whose stored `tenant_id` ≠ requester's is dropped and alerts; kill-switch disables cache |
| 8 | **Indirect prompt injection in retrieved chunk** | Hijacked agent goal | 5 | 3 | 2 | 30 | `injection_shield` on chunks, goal-drift score (KPI-18) | Quarantine chunk; strip instructions; GoalAnchorBlock re-asserted; Execution subagent unreachable from Retrieval output; Tier 3 still needs human approval |
| 9 | **Runaway tool loop / repeated invalid args** | Latency, cost, wrong action | 4 | 3 | 1 | 12 | Same (tool, args-hash) failed ×2 | `HALTED_BREAKER` with remediation prompt; hard cap 5 iterations / 450 ms |
| 10 | **PII regex/model miss** (e.g., `name[at]domain`) | PII leaks into logs/prompt | 5 | 3 | 3 | 45 | Recall audit (KPI-02), red-team suite | Redactor is **fail-closed**; secondary regex pass on egress; logs store hashes only; weekly recall audit on new obfuscation patterns |
| 11 | **Embedding model / index drift** (stale or mismatched index) | Low recall, wrong chunks | 3 | 2 | 3 | 18 | Nightly recall@50 canary, index version tag mismatch | Refuse to serve if query-time model version ≠ index version; rebuild; serve last-known-good index snapshot |
| 12 | **Approval service/UI down or approver unavailable** | Tier 3 requests stuck | 3 | 3 | 1 | 9 | Pending-age alert | Requests stay `PENDING_APPROVAL` (never auto-approve); expire after 24 h to `EXPIRED`; notify requester; no autonomous execution under any failure |

Highest-priority items by RPN: #10 (45), #6 (30), #8 (30). They get extra tests in the red-team and chaos suites.

---

## 7. Architecture Decision Records

### ADR-1: Hybrid retrieval weights (BM25 + dense via RRF)
- **Status:** Accepted
- **Context:** Single-keyword BM25 was blind to paraphrase; dense-only misses exact identifiers (SKUs, clause numbers). The baseline had a 38.6% hallucination rate.
- **Decision:** Run BM25 (k1=1.25, b=0.75, top 50) and dense bi-encoder (`bge-large-en-v1.5`, FAISS HNSW, efSearch=64, top 50) **in parallel**, fuse with Reciprocal Rank Fusion (k=60, dedupe by `chunk_hash`, pool 40), then rerank with a cross-encoder (top 4, sigmoid > 0.42).
- **Why RRF, not weighted-score fusion:** BM25 scores and cosine scores are on different scales; RRF uses ranks only, needs no calibration, and is robust across corpora. Equal rank weighting is the default; per-lane weights are exposed for tuning but default to 1.0 each.
- **Consequences:** Two retrieval lanes cost extra compute but run in parallel (critical path = max). The 0.42 threshold is a starting value; it must be tuned on the golden set and may move (the incident runbook raises it to 0.55 under hallucination surges).
- **Revisit when:** context recall@50 < 0.92 or precision@4 < 0.88.

### ADR-2: ONNX quantization for embedding and reranking
- **Status:** Accepted
- **Context:** Rerank budget is 45 ms and embedding 25 ms at high concurrency; full-precision PyTorch inference is too slow and memory-heavy.
- **Decision:** Export `bge-large-en-v1.5` and `bge-reranker-v2-m3` to ONNX. Use **FP16** on GPU and **INT8** (dynamic quantization) on CPU fallback. Dynamic micro-batching: batch 32, max wait 8 ms.
- **Alternatives rejected:** PyTorch eager (latency); TensorRT-only (portability; kept as roadmap).
- **Consequences:** Quantization can reduce ranking quality slightly. A release gate compares nDCG@4 and MRR@4 of quantized vs. full-precision models on the golden set; a drop of > 1 point blocks the rollout. The 8 ms batching wait is counted in the stage budgets.
- **Revisit when:** quality gate fails, or GPU capacity changes.

### ADR-3: Redis key isolation and cache design
- **Status:** Accepted
- **Context:** The baseline used global keys; two users with the same query shared cached answers, leaking data across users.
- **Decision:** Key = `cache:{tenant_id}:` + `SHA256(tenant_id:user_id:normalized_query)`. Semantic matching (cosine ≥ 0.92) is restricted to entries in the **same tenant and user scope**. TTL 300 s ± 45 s jitter to prevent stampedes. Idempotency keys and breaker counters use the same `{tenant_id}` prefix. Redis Cluster hash tags keep one tenant's keys on one slot group.
- **Consequences:** Lower hit ratio than a shared cache (target 35% in steady state, KPI-27) in exchange for zero cross-tenant leakage (KPI-28, must be 100%). Only responses that passed Verification are cached (FMEA #6).
- **Follow-up:** If the LLM p95 does not fit the 250 ms allowance (Section 4.6), consider a smaller model or a higher-hit semantic cache, still tenant-scoped.

### ADR-4: Bounded ReAct loop with deterministic state machine
- **Status:** Accepted
- **Context:** Unbounded ReAct loops retried on minor typos (`SKU99`) indefinitely.
- **Decision:** State machine `INIT → TRIAGE → REASONING → TOOL_CALL → OBSERVATION → VERIFY → DONE | HALTED`. Limits: `max_iterations=5`, `per_step_timeout=120 ms`, `total_loop_budget=450 ms`, no recursive self-calls without state mutation. Harness auto-repairs known formats before counting a failure. Circuit breaker: the same `(tool, canonical_args_hash)` failing twice → `HALTED_BREAKER`. Immutable `GoalAnchorBlock` at index 0; observations compressed to ≤ 180 tokens; last 3 turns kept hot, verified facts in Postgres.
- **Alternatives rejected:** Free-form agent loops (unbounded cost/latency); a single LLM call with no tools (cannot handle transactions or multi-hop).
- **Consequences:** Some hard queries will halt rather than continue; users get a structured remediation prompt. Mean steps target ≤ 2.2 (KPI-17); 100% loop escape (KPI-16).
- **Revisit when:** the halted-task rate exceeds an agreed threshold on the golden set.

### ADR-5: Human approval is an architectural invariant
- **Status:** Accepted
- **Decision:** `submit_transaction` can only create `PENDING_APPROVAL`; execution is performed by a separate audited service after a different human approves. No flag, prompt, or tool argument can bypass this. Failure of any dependency fails closed.
- **Consequences:** Transactions are never real-time; KPI-19 (100% enforcement) is verified by audit-log integrity checks.

---

## Change log
| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-11-18 | Initial design, locked. |
| 1.1 | 2026-11-18 | Corrected Section 4 arithmetic after read-through: critical-path retrieval is 77 ms (not 102); pre-LLM total is 132 ms (not 122); p95 row in 4.3 changed 122 → 132 so rows sum to 450. Budget totals (450 / 600) unchanged. KPI thresholds verified 36/36 against Blueprint v2. |
| 1.2 | 2026-11-18 | Prompt 4 build. Chunk resource URI is now `rag://chunks/{tenant_id}/{chunk_id}` (was `rag://chunks/{id}`) so tenant isolation is enforceable on reads. Added error code behaviour: unknown tenant index -> `E_TENANT_MISMATCH`; stale index -> `E_UPSTREAM_UNAVAILABLE`. Dev/test models are stand-ins: `bge-small-en-v1.5` (embedder) and `ms-marco-MiniLM-L-6-v2` (reranker); the 0.42 threshold and the 25 ms / 45 ms stage budgets (KPI-08/09) are NOT met on the 4-vCPU CPU dev box and must be re-measured on target hardware (GPU, FP16). |
| 1.3 | 2026-11-18 | Prompt 5 build. Implemented `E_NOT_GROUNDED` (-32005). State server denies Tier 3 unless the instance is started with `STATE_SERVER_MAX_TIER=3` (no per-caller identity until Prompt 8). Database app role is SELECT/INSERT only (+UPDATE on `session_facts`), with FORCE row-level security; audit rows are append-only and store digests, never raw values. Cache semantic scope is tenant AND user (per ADR-3). |
| 1.4 | 2026-11-18 | Prompt 6 build (triage + bounded ReAct). Triage: rules + nearest-neighbour over bge-small embeddings, `min_sim=0.65` chosen on a DEV split; held-out TEST 56/57 (98.2%), p95 14.7 ms. Triage `domain` is advisory only and must not be used as a hard retrieval filter (one held-out wire-approval question got the right route but the wrong domain). Harness: cancelled calls now release their idempotency key. **OPEN ISSUE:** section 4 budgets 250 ms for LLM generation to the last token, but ADR-4 sets `per_step_timeout = 120 ms` for every reasoning step including the one that emits the final answer. Both cannot hold with a real LLM; the loop currently applies 120 ms to every reasoning call. Decide before wiring a real model: either give the final-answer step its own limit (remaining budget) or raise the per-step limit. |
| 1.5 | 2026-11-18 | Prompt 7 build (subagents, bulkheads, abstention, verification). Isolation is structural: `ScopedHarness` = per-subagent tool allow-list + tier ceiling (Retrieval T1 read-only, Execution T3 state-store only, Verification no tools). Added a 4th bulkhead pool `triage` (16) so routing is never queued behind retrieval; sizes rag=64, db=128, llm=256 as designed. Verification is layered (citation check, number guard, absolute-word guard, NLI `nli-deberta-v3-xsmall` over the best sentences) because NLI alone accepted an unsupported claim. **INCONSISTENCY:** section 4.4 gives the abstention path a 90 ms target while also saying retrieval completes first; but hooks (15) + cache (5) + triage (25) + retrieval (~77) already exceed 90 ms. Implemented reading: the 90 ms covers only the decision after retrieval returns (measured 0 ms); the end-to-end abstention path is retrieval-bound (~300 ms warm on this CPU). Decide whether 4.4 should read ~125 ms on target hardware. **MEASURED (CPU, warm):** answered question ~400-490 ms with a no-LLM extractive answerer, so the 450 ms p95 budget is already consumed before a real LLM; GPU serving is required to meet section 4. Claim verification ~115 ms per claim vs the 68 ms reserve. |
| 1.6 | 2026-11-18 | Prompt 8 build (plugins, governance). `mcp_config.json` now drives plugin priority / stages / fail_closed. Fail-closed plugins (pii_redactor, injection_shield, hitl_gate) cannot be toggled at runtime and the pipeline refuses to start without them; only non-critical plugins hot-toggle, and every toggle is audited under tenant `platform-ops`. Added a third hook `scan_retrieved` (indirect-injection scan of retrieved chunks, quarantine then abstain). Added error code `E_PII_DETECTED` (-32007). Raw-PII persistence guards on facts, transaction justifications, audit payloads and the semantic cache. Manifests: Ed25519-signed, hosts hold only the public key, live tool contracts are compared against the signed manifest. **MEASURED:** KPI-01 pre-hook p95: 1.1 ms (regex PII + patterns), 12.0 ms (+ Presidio NER) = meets 15 ms; +semantic injection layer adds ~15-20 ms (31.8 ms full) = misses. Recommended default: NER + patterns, semantic opt-in until it can reuse triage's embedding. PII held-out 22/22 redacted, 0/16 clean texts altered; injection held-out 21/21 blocked, 0/47 benign flagged, BUT evasive paraphrase probes: patterns 1/8, + semantic 5/8, so KPI-03 (>= 99.5%) is NOT demonstrated; structural controls are the real defence. **OPEN:** (1) the state server still has no per-caller identity: Tier 3 stays default-deny per instance; (2) no approver decision API or post-approval executor exists yet (needed by the Prompt 13 approval queue; the app DB role deliberately cannot UPDATE approvals); (3) session circuit breaker is in-process only. |
| 1.7 | 2026-11-18 | Prompt 9 build (Obsidian vault + Graphify). The blueprint's `graphify` CLI does not exist as a public package; a self-contained graph engine was built (`mcp_servers/graphify.py`, no networkx). Vault: 40 atomic notes (`knowledge_vault/atomic/`, one rule each, verbatim from its policy so chunk citations and NLI verification still hold) + 12 policy hubs; graph.json = 97 nodes (notes, policies, 32 modules, 13 Pydantic schemas), 0 dangling links. New tool `graphify_subgraph` (T1, signed manifest) and a `token_counter` plugin (scan_retrieved hook) that logs context tokens before/after. Orchestrator `graph_mode`: off (default) | measure | use. **MEASURED over 100 queries (dense anchoring, 1 anchor, 320-token budget):** answer coverage 99% (anchor is a gold note 99%; the one miss is a two-policy question with no link between the policies). Tokens: vs a naive top-4-chunks prompt 228.5 -> 70.2 = **69.3% reduction (KPI-26 target >= 45% met)**; BUT vs the chunks this pipeline actually keeps (score > 0.42; mean 53.7 tokens) the graph is **30.6% LARGER**, and vs today's 180-token compressed observation only 3.5% smaller. Cause: this corpus's policy chunks are ~50 tokens, so there is little to compress; savings need long chunks (~512 tokens) and have NOT been measured on such a corpus. Therefore graph context is opt-in, not default. KPI-26 is reported against the naive baseline only, with this caveat. **Anchoring:** BM25 alone 95% coverage (vocabulary mismatches: MFA/multi-factor, dormant/unused); dense fusion 99%; a second anchor did not help and cost tokens. The design's 320-token subgraph budget exceeds the 180-token observation cap in section 4/ADR-4 (a graph context shown to the model is exempt from the cap); decide whether that is intended. |
| 1.8 | 2026-11-18 | Prompt 10 build (API, tracing, observability). `app/main.py` (FastAPI): POST /v1/answer (JSON or SSE), GET /healthz (no bulkhead slot, no I/O), GET /readyz. Caller identity at the edge: HMAC-signed bearer tokens carry tenant, user and a TIER CEILING (`app/auth.py`); the request body cannot name a tenant/user/tier (unknown fields are 422). This closes the API-edge part of the Prompt 8 'caller identity' gap; the standalone state-store MCP server still fixes its tier per instance. Pipeline order is now ingress -> auth -> pre-plugins -> cache -> triage -> subagents -> post-plugins -> background cache fill, with the cache keyed on the REDACTED query and filled only from verified HYBRID_RAG answers. Tool handlers are hosted IN-PROCESS via the same registrar functions the MCP servers use (one implementation, no MCP hop): KPI-30 (MCP serialization overhead) is therefore NOT exercised by the API, and the API does not enforce signed manifests at runtime. Spans: all ten required names exist (`hook.pre`, `cache.lookup`, `triage.classify`, `mcp.rag.bm25/biencoder/rrf/rerank`, `agent.react.step`, `llm.generate`, `hook.post.ragas`) and were verified in ONE trace in Tempo, continuing an incoming W3C `traceparent`. `hook.post.ragas` is a background plugin computing PROXIES (context utilisation, query/answer cosine), not RAGAS metrics. Streaming: no LLM token stream exists, so SSE sends the finished answer in chunks (first event is immediate; first answer chunk waits for the pipeline). **BUGS FOUND AND FIXED THIS PROMPT:** (1) bulkhead `run_in_executor` dropped contextvars, so every span created inside a pooled call started a NEW trace (since Prompt 7); (2) the latency metric skipped cache hits and blocked requests; (3) triage routed policy QUESTIONS about transfers (e.g. 'Who has to approve a wire transfer above 10,000 USD?', also with a preamble sentence) to TRANSACTION_EXEC; (4) the HITL gate treated a clarification on a transaction route as a violation, so a new outcome NEEDS_INFO was added. Fix (3) came from an end-to-end test, not from the held-out set, so that set is no longer untouched (accuracy unchanged at 98.2%). |
| 1.9 | 2026-11-18 | Golden set grown 250 -> 273: 23 new plausible-but-uncovered negatives (authored without viewing TEST results; split stays 40/60 by position). Amends 5.1 strata. File name `golden_250.jsonl` is kept to avoid churn. Abstention threshold is still calibrated on DEV only. |
| 1.10 | 2026-11-18 | Prompt 12 build (red team + chaos). `evals/redteam.py` (100 attacks, 100 benign controls; loopback-only `--url`), `evals/chaos.py` (5 scenarios). Run in-process because Postgres/Redis were not available; cross-tenant probe used `evals/fake_redis.py`, so it proves application scoping only. Bug found and fixed: `repair_amount` raised `decimal.InvalidOperation` on a 40-digit amount. **OPEN, measured:** KPI-03 injection block rate 66.7% vs 99.5% (direct 13/25, indirect 17/20); KPI-06 false positives 1/112 (0.9%, passes). Crescendo 7/10, PII smuggling 17/20 contained. **OPEN gaps vs FMEA:** a dead idempotency store raises `ConnectionError` out of the harness (#1); a reasoner 429 escapes the ReAct loop (#4). KPI-16 100% (33/33), cross-tenant leaks 0, MCP hang bounded at ~240 ms. |

"""Tool contracts: Pydantic v2 models for the four MCP tools.

Implements docs/design.md v1.1, Section 2 (regex constraints, strict models).
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

TENANT_RE = r"^[a-z0-9][a-z0-9_-]{2,39}$"
USER_RE = r"^[A-Za-z0-9_.-]{1,64}$"
UUID4_RE = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
FACT_KEY_RE = r"^[a-z][a-z0-9_.]{2,63}$"
CHUNK_ID_RE = r"^[a-z0-9]{2,16}-[0-9]{4}-[0-9a-f]{8}$"
SKU_RE = r"^SKU-\d{3}$"
LOT_RE = r"^LOT-\d{5}$"
ACCOUNT_RE = r"^ACC-\d{8}$"
CURRENCY_RE = r"^[A-Z]{3}$"
AMOUNT_RE = r"^\d{1,12}(\.\d{1,2})?$"

Tenant = Annotated[str, StringConstraints(pattern=TENANT_RE)]
UserId = Annotated[str, StringConstraints(pattern=USER_RE)]
UUID4 = Annotated[str, StringConstraints(pattern=UUID4_RE)]
FactKey = Annotated[str, StringConstraints(pattern=FACT_KEY_RE)]


class ToolInputBase(BaseModel):
    """Every tool input carries tenant, user and an idempotency key (design B3, B2)."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    tenant_id: Tenant
    user_id: UserId
    request_id: UUID4


# --- hybrid_retrieve (TIER_1_READ) ------------------------------------------
class HybridRetrieveIn(ToolInputBase):
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    top_k_sparse: int = Field(50, ge=1, le=100)
    top_k_dense: int = Field(50, ge=1, le=100)
    rrf_k: int = Field(60, ge=1, le=200)
    rerank_k: int = Field(4, ge=1, le=10)
    score_threshold: float = Field(0.42, ge=0.0, le=1.0)
    domain: str | None = Field(None, pattern=r"^[a-z][a-z0-9_]{1,31}$")


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(pattern=CHUNK_ID_RE)
    text: str
    score: float = Field(ge=0, le=1)
    bm25_rank: int | None
    dense_rank: int | None
    source_uri: str


class HybridRetrieveOut(BaseModel):
    abstain: bool
    chunks: list[RetrievedChunk]
    top_score: float
    stage_latency_ms: dict[Literal["bm25", "dense", "rrf", "rerank"], float]


# --- get_session_facts (TIER_1_READ) ----------------------------------------
class GetSessionFactsIn(ToolInputBase):
    session_id: UUID4
    keys: list[FactKey] | None = Field(None, max_length=50)
    limit: int = Field(50, ge=1, le=200)


class SessionFact(BaseModel):
    fact_key: str
    value: str
    verified: bool
    created_at: str


class GetSessionFactsOut(BaseModel):
    facts: list[SessionFact]


# --- write_session_fact (TIER_2_COMPUTE) ------------------------------------
class WriteSessionFactIn(ToolInputBase):
    session_id: UUID4
    fact_key: FactKey
    value: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    source_chunk_ids: list[Annotated[str, StringConstraints(pattern=CHUNK_ID_RE)]] = Field(
        min_length=1, max_length=8
    )
    verified: Literal[True]


class WriteSessionFactOut(BaseModel):
    stored: bool
    fact_key: str
    audit_hash: str


# --- submit_transaction (TIER_3_MUTATE) -------------------------------------
class SubmitTransactionIn(ToolInputBase):
    session_id: UUID4
    action: Literal["TRANSFER", "ORDER", "STATUS_CHANGE"]
    account_id: str = Field(pattern=ACCOUNT_RE)
    amount: str | None = Field(None, pattern=AMOUNT_RE)
    currency: str | None = Field(None, pattern=CURRENCY_RE)
    justification: Annotated[str, StringConstraints(min_length=10, max_length=500)]


class SubmitTransactionOut(BaseModel):
    status: Literal["PENDING_APPROVAL"]  # the ONLY status this tool may return
    approval_id: UUID4
    audit_hash: str


# --- graphify_subgraph (TIER_1_READ) ----------------------------------------
class GraphifySubgraphIn(ToolInputBase):
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    depth: int = Field(2, ge=1, le=2)                       # design Gate 17: at most 2 hops
    token_budget: int = Field(320, ge=50, le=1000)
    hub_ids: list[Annotated[str, StringConstraints(pattern=r"^[a-z0-9]{2,16}$")]] | None = Field(None, max_length=8)


class GraphNodeOut(BaseModel):
    id: str
    type: Literal["note", "policy", "module", "schema"]
    title: str
    text: str
    hub: str | None
    domain: str | None
    risk_tier: int
    distance: int = Field(ge=0, le=2)
    relevance: float
    bridge: bool


class GraphEdgeOut(BaseModel):
    src: str
    dst: str
    type: str


class GraphifySubgraphOut(BaseModel):
    anchor: str | None
    anchor_score: float
    anchors: list[str]
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
    rendered: str
    tokens: int

"""Stateless ONNX model worker (embedding + rerank) with dynamic micro-batching. Scale it horizontally behind a Service/HPA.
  uvicorn mcp_servers.model_worker:app --port 8090
POST /embed  {"texts": [...], "is_query": bool}      POST /rerank {"query": str, "passages": [...]}
Embedding requests from different callers are coalesced (batch 32, max_wait 8 ms). Rerank requests are batched per query by the
model itself (BATCH=32 inside OnnxReranker), so they are queued one-per-request and run serially on one session per worker.
No tenant data is stored; callers must not send raw PII (the API redacts before retrieval)."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.microbatch import MicroBatcher
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, DEFAULT_RERANK_DIR, OnnxEmbedder, OnnxReranker

MAX_BATCH = int(os.environ.get("MB_MAX_BATCH", "32"))
MAX_WAIT_MS = float(os.environ.get("MB_MAX_WAIT_MS", "8"))


class EmbedIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    is_query: bool = False


class RerankIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    passages: list[str] = Field(min_length=1, max_length=64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    emb = OnnxEmbedder(os.environ.get("EMBED_DIR", DEFAULT_EMBED_DIR), threads=2)
    rer = OnnxReranker(os.environ.get("RERANK_DIR", DEFAULT_RERANK_DIR), threads=2)

    def embed_batch(items):  # items: list[(text, is_query)] -> list[list[float]]; split by is_query so prefixes stay correct
        out = [None] * len(items)
        for flag in (True, False):
            idx = [i for i, (_, q) in enumerate(items) if q is flag]
            if idx:
                vecs = emb.encode([items[i][0] for i in idx], is_query=flag)
                for i, v in zip(idx, vecs):
                    out[i] = v.tolist()
        return out

    app.state.embed = MicroBatcher(embed_batch, MAX_BATCH, MAX_WAIT_MS, "embed")
    app.state.rerank = MicroBatcher(lambda reqs: [rer.score(q, p).tolist() for q, p in reqs], 1, 0.0, "rerank")
    yield
    await app.state.embed.close()
    await app.state.rerank.close()


app = FastAPI(title="model-worker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/embed")
async def embed(m: EmbedIn):
    import asyncio
    try:
        return {"vectors": await asyncio.gather(*[app.state.embed.submit((t, m.is_query)) for t in m.texts])}
    except asyncio.QueueFull:
        raise HTTPException(429, "worker saturated", headers={"Retry-After": "1"}) from None


@app.post("/rerank")
async def rerank(m: RerankIn):
    import asyncio
    try:
        return {"scores": await app.state.rerank.submit((m.query, m.passages))}
    except asyncio.QueueFull:
        raise HTTPException(429, "worker saturated", headers={"Retry-After": "1"}) from None

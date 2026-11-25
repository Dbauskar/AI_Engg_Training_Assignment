"""3-stage hybrid retrieval (docs/design.md v1.1, Gates 3 & 8).

  Stage 1  BM25Okapi (k1=1.25, b=0.75, porter stemming, top 50)  ||  dense bi-encoder + FAISS HNSW (top 50)
  Fusion   Reciprocal Rank Fusion (k=60, dedupe by chunk_hash, pool 40)
  Stage 3  cross-encoder rerank (ONNX, batch 32), keep top rerank_k with sigmoid score > threshold

Models run through onnxruntime (CPU here). Defaults are the small stand-ins (bge-small-en-v1.5 and
ms-marco-MiniLM-L-6-v2); the ONNX loader is model-agnostic, so the design's full-size models are a
config change, not a code change.

Privacy: query text is never put in spans or logs (only its length).
Isolation: one index directory per tenant; tenant_id is regex-validated before touching the filesystem.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import onnxruntime as ort
from nltk.stem import PorterStemmer
from opentelemetry import context as otel_context
from opentelemetry import trace
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

from app.contracts import CHUNK_ID_RE, TENANT_RE, HybridRetrieveOut, RetrievedChunk

_tracer = trace.get_tracer("unreliable_assistant.rag")

# --- design constants --------------------------------------------------------
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64
BM25_K1 = 1.25
BM25_B = 0.75
TOP_K = 50
RRF_K = 60
POOL = 40
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64
BATCH = 32
MAX_LEN = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_EMBED_DIR = "models/bge-small-en-v1.5"
DEFAULT_RERANK_DIR = "models/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL_ID = "bge-small-en-v1.5"
RERANK_MODEL_ID = "ms-marco-MiniLM-L-6-v2"

_TENANT = re.compile(TENANT_RE)
_CHUNK_ID = re.compile(CHUNK_ID_RE)


class IndexNotFound(Exception):
    pass


class IndexVersionError(Exception):
    pass


# --- text utilities -----------------------------------------------------------
_STOP = frozenset(
    "a an and are as at be by for from has have how i in is it of on or that the this to was what when "
    "which who will with do does we our you your can".split()
)
_stemmer = PorterStemmer()


@lru_cache(maxsize=65536)
def _stem(w: str) -> str:
    return _stemmer.stem(w)


def bm25_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stop-words removed, Porter-stemmed ('SKU-099' -> ['sku','099'])."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_note(raw: str) -> tuple[dict[str, str], str]:
    """Split YAML-ish frontmatter (simple `key: value` lines) from the body."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def chunk_text(text: str, tok: Tokenizer, size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding window over tokenizer tokens (size, overlap). Chunks are cut on original-text offsets."""
    text = text.strip()
    if not text:
        return []
    enc = tok.encode(text, add_special_tokens=False)
    n = len(enc.ids)
    if n <= size:
        return [text]
    step = size - overlap
    out, start = [], 0
    while True:
        end = min(start + size, n)
        out.append(text[enc.offsets[start][0]: enc.offsets[end - 1][1]].strip())
        if end >= n:
            return out
        start += step


# --- ONNX models -------------------------------------------------------------
class _Onnx:
    def __init__(self, model_dir: str | Path, max_len: int = MAX_LEN, threads: int | None = None) -> None:
        d = Path(model_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()  # pad to longest in the batch (dynamic padding)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or os.cpu_count() or 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(d / "onnx" / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self._names = {i.name for i in self.sess.get_inputs()}

    def _run(self, encodings: list[Any]) -> np.ndarray:
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        return self.sess.run(None, feed)[0]


class OnnxEmbedder(_Onnx):
    """CLS-pooled, L2-normalised sentence embeddings (BGE convention)."""

    model_id = EMBED_MODEL_ID

    _MEMO_MAX = 512

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self._memo: dict[str, np.ndarray] = {}  # per instance: single-text, non-query encodes (triage, cache and injection classifier share them)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if len(texts) == 1 and not is_query:
            hit = self._memo.get(texts[0])
            if hit is not None:
                return hit.copy()
            out = self._encode(texts, False)
            try:
                if len(self._memo) >= self._MEMO_MAX:
                    self._memo.pop(next(iter(self._memo)), None)
                self._memo[texts[0]] = out.copy()
            except (RuntimeError, StopIteration):  # concurrent mutation: the memo is only an optimisation
                pass
            return out
        return self._encode(texts, is_query)

    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        outs = []
        for i in range(0, len(texts), BATCH):
            hidden = self._run(self.tok.encode_batch(texts[i: i + BATCH]))
            v = hidden[:, 0, :]
            outs.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.ascontiguousarray(np.vstack(outs), dtype=np.float32)


class OnnxReranker(_Onnx):
    """Cross-encoder: sigmoid(logit) relevance score per (query, passage) pair."""

    model_id = RERANK_MODEL_ID

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))  # similar lengths -> less padding
        scores = np.zeros(len(passages), dtype=np.float32)
        for i in range(0, len(order), BATCH):
            idx = order[i: i + BATCH]
            enc = self.tok.encode_batch([(query, passages[j]) for j in idx])
            logits = self._run(enc).reshape(len(idx), -1)[:, 0]
            scores[idx] = 1.0 / (1.0 + np.exp(-logits))
        return scores


# --- index -------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    domain: str
    seq: int
    text: str
    text_hash: str
    source_uri: str


@dataclass
class TenantIndex:
    chunks: list[Chunk]
    bm25: BM25Okapi
    faiss_index: Any
    by_id: dict[str, Chunk]


def _alnum_doc_id(doc_id: str) -> str:
    return re.sub(r"[^a-z0-9]", "", doc_id.lower())[:16]


def build_index(tenant_id: str, src_dir: str | Path, out_root: str | Path,
                embedder: OnnxEmbedder, chunk_tok: Tokenizer) -> Path:
    if not _TENANT.match(tenant_id):
        raise ValueError("invalid tenant_id")
    src, out = Path(src_dir), Path(out_root) / tenant_id
    chunks: list[Chunk] = []
    for path in sorted(src.glob("*.md")):
        meta, body = parse_note(path.read_text(encoding="utf8"))
        doc_id = _alnum_doc_id(meta.get("id", path.stem)) or "doc"
        for seq, text in enumerate(chunk_text(body, chunk_tok)):
            h = sha(text)
            chunks.append(Chunk(f"{doc_id}-{seq:04d}-{h[:8]}", doc_id, meta.get("title", path.stem),
                                meta.get("domain", "general"), seq, text, h, path.name))
    if not chunks:
        raise ValueError(f"no .md content found under {src}")
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk_id (duplicate document ids or content)")
    vecs = embedder.encode([c.text for c in chunks])
    idx = faiss.IndexHNSWFlat(vecs.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.add(vecs)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(out / "faiss.index"))
    (out / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1))
    (out / "manifest.json").write_text(json.dumps({
        "tenant_id": tenant_id, "embedding_model": embedder.model_id, "dim": int(vecs.shape[1]),
        "chunks": len(chunks), "chunk_tokens": CHUNK_TOKENS, "overlap": CHUNK_OVERLAP}))
    return out


def load_index(tenant_id: str, root: str | Path, embedder_id: str) -> TenantIndex:
    if not _TENANT.match(tenant_id):  # also blocks path traversal ('..', '/')
        raise IndexNotFound("invalid tenant_id")
    d = Path(root) / tenant_id
    if not (d / "manifest.json").exists():
        raise IndexNotFound("no index for tenant")
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["embedding_model"] != embedder_id:  # FMEA #11: refuse stale/mismatched index
        raise IndexVersionError("index was built with a different embedding model; rebuild required")
    chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
    fi = faiss.read_index(str(d / "faiss.index"))
    fi.hnsw.efSearch = EF_SEARCH
    bm25 = BM25Okapi([bm25_tokens(c.text) for c in chunks], k1=BM25_K1, b=BM25_B)
    return TenantIndex(chunks, bm25, fi, {c.chunk_id: c for c in chunks})


# --- fusion ------------------------------------------------------------------
def rrf_fuse(rank_lists: list[list[int]], hashes: list[str], k: int = RRF_K, pool: int = POOL) -> list[int]:
    """Reciprocal Rank Fusion over lists of chunk indices (best first). Dedupes by content hash."""
    score: dict[int, float] = {}
    for lst in rank_lists:
        for r, i in enumerate(lst, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + r)
    out, seen = [], set()
    for i in sorted(score, key=lambda j: (-score[j], j)):
        if hashes[i] in seen:
            continue
        seen.add(hashes[i])
        out.append(i)
        if len(out) == pool:
            break
    return out


# --- retriever ---------------------------------------------------------------
class HybridRetriever:
    def __init__(self, index_root: str | Path = "data/index", embed_dir: str | Path = DEFAULT_EMBED_DIR,
                 rerank_dir: str | Path = DEFAULT_RERANK_DIR, threads: int | None = None,
                 lane_workers: int = 128, embedder: OnnxEmbedder | None = None) -> None:
        self.root = Path(index_root)
        threads = threads or 2  # measured best on 4 vCPU: more threads contend
        self.embedder = embedder or OnnxEmbedder(embed_dir, threads=threads)
        self.reranker = OnnxReranker(rerank_dir, threads=threads)
        self._indexes: dict[str, TenantIndex] = {}
        # 2 lanes per in-flight request: must be >= 2 x the rag bulkhead (64), or concurrent requests serialise here
        self._pool = ThreadPoolExecutor(max_workers=lane_workers, thread_name_prefix="rag-lane")

    def index_for(self, tenant_id: str) -> TenantIndex:
        if tenant_id not in self._indexes:
            self._indexes[tenant_id] = load_index(tenant_id, self.root, self.embedder.model_id)
        return self._indexes[tenant_id]

    def get_chunk(self, tenant_id: str, chunk_id: str) -> Chunk:
        if not _CHUNK_ID.match(chunk_id):
            raise IndexNotFound("invalid chunk_id")
        ch = self.index_for(tenant_id).by_id.get(chunk_id)
        if ch is None:
            raise IndexNotFound("chunk not found")  # same answer for other tenants' ids: no oracle
        return ch

    # -- stages (each emits its own span and returns its latency) --
    def _bm25(self, ti: TenantIndex, query: str, k: int, domain: str | None, ctx: Any):
        tok = otel_context.attach(ctx)
        try:
            with _tracer.start_as_current_span("mcp.rag.bm25") as sp:
                t0 = time.perf_counter()
                scores = ti.bm25.get_scores(bm25_tokens(query))
                order = np.argsort(-scores)[: max(k * 2, k)]
                hits = [int(i) for i in order if scores[i] > 0 and (domain is None or ti.chunks[i].domain == domain)][:k]
                ms = (time.perf_counter() - t0) * 1000
                sp.set_attribute("rag.candidates", len(hits))
                return hits, ms
        finally:
            otel_context.detach(tok)

    def _dense(self, ti: TenantIndex, query: str, k: int, domain: str | None, ctx: Any):
        tok = otel_context.attach(ctx)
        try:
            with _tracer.start_as_current_span("mcp.rag.biencoder") as sp:
                t0 = time.perf_counter()
                q = self.embedder.encode([query], is_query=True)
                _, ids = ti.faiss_index.search(q, min(max(k * 2, k), len(ti.chunks)))
                hits = [int(i) for i in ids[0] if i >= 0 and (domain is None or ti.chunks[i].domain == domain)][:k]
                ms = (time.perf_counter() - t0) * 1000
                sp.set_attribute("rag.candidates", len(hits))
                return hits, ms
        finally:
            otel_context.detach(tok)

    def candidate_docs(self, tenant_id: str, query: str, k: int = TOP_K) -> dict[str, list[str]]:
        """Document ids (best first, de-duplicated) that each stage would hand on. For evaluation/diagnosis only."""
        ti = self.index_for(tenant_id)
        ctx = otel_context.get_current()
        bm, _ = self._bm25(ti, query, k, None, ctx)
        de, _ = self._dense(ti, query, k, None, ctx)
        pool = rrf_fuse([bm, de], [c.text_hash for c in ti.chunks], RRF_K, POOL)
        doc = lambda idxs: list(dict.fromkeys(ti.chunks[i].doc_id for i in idxs))  # noqa: E731
        return {"bm25": doc(bm), "dense": doc(de), "union": doc(bm + de), "pool": doc(pool)}

    def retrieve(self, tenant_id: str, query: str, *, top_k_sparse: int = TOP_K, top_k_dense: int = TOP_K,
                 rrf_k: int = RRF_K, rerank_k: int = 4, score_threshold: float = 0.42,
                 domain: str | None = None) -> HybridRetrieveOut:
        ti = self.index_for(tenant_id)
        with _tracer.start_as_current_span("mcp.rag.hybrid_retrieve") as root:
            root.set_attribute("tenant.id", tenant_id)
            root.set_attribute("rag.query_len", len(query))
            ctx = otel_context.get_current()
            f_bm = self._pool.submit(self._bm25, ti, query, top_k_sparse, domain, ctx)
            f_de = self._pool.submit(self._dense, ti, query, top_k_dense, domain, ctx)
            bm_hits, bm_ms = f_bm.result()
            de_hits, de_ms = f_de.result()

            with _tracer.start_as_current_span("mcp.rag.rrf") as sp:
                t0 = time.perf_counter()
                pool = rrf_fuse([bm_hits, de_hits], [c.text_hash for c in ti.chunks], rrf_k, POOL)
                rrf_ms = (time.perf_counter() - t0) * 1000
                sp.set_attribute("rag.pool", len(pool))

            with _tracer.start_as_current_span("mcp.rag.rerank") as sp:
                t0 = time.perf_counter()
                scores = self.reranker.score(query, [ti.chunks[i].text for i in pool])
                rr_ms = (time.perf_counter() - t0) * 1000
                sp.set_attribute("rag.reranked", len(pool))

            ranked = sorted(zip(pool, scores), key=lambda p: -float(p[1]))
            top_score = float(ranked[0][1]) if ranked else 0.0
            keep = [(i, s) for i, s in ranked if s > score_threshold][:rerank_k]
            abstain = not keep
            root.set_attribute("rag.abstain", abstain)
            root.set_attribute("rag.top_score", top_score)
            bm_rank = {i: r for r, i in enumerate(bm_hits, 1)}
            de_rank = {i: r for r, i in enumerate(de_hits, 1)}
            chunks = [RetrievedChunk(chunk_id=ti.chunks[i].chunk_id, text=ti.chunks[i].text, score=float(s),
                                     bm25_rank=bm_rank.get(i), dense_rank=de_rank.get(i),
                                     source_uri=ti.chunks[i].source_uri) for i, s in keep]
            return HybridRetrieveOut(
                abstain=abstain, chunks=chunks, top_score=top_score,
                stage_latency_ms={"bm25": bm_ms, "dense": de_ms, "rrf": rrf_ms, "rerank": rr_ms})

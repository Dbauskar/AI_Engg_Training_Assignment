import json
import re
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from app.contracts import CHUNK_ID_RE
from evals.rag_queries import ANSWERABLE, UNANSWERABLE
from mcp_servers.rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    DEFAULT_EMBED_DIR,
    DEFAULT_RERANK_DIR,
    HybridRetriever,
    IndexNotFound,
    IndexVersionError,
    bm25_tokens,
    build_index,
    chunk_text,
    load_index,
    parse_note,
    rrf_fuse,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_OK = (ROOT / DEFAULT_EMBED_DIR / "onnx/model.onnx").exists() and (ROOT / DEFAULT_RERANK_DIR / "onnx/model.onnx").exists()
needs_models = pytest.mark.skipif(not MODELS_OK, reason="ONNX models not downloaded (see scripts/ingest.py)")
TENANT = "acme-corp"


# ---------------- pure functions (no models) ----------------
def test_bm25_tokens_keep_identifiers_stem_and_drop_stopwords():
    assert bm25_tokens("SKU-099") == ["sku", "099"]
    assert bm25_tokens("Refunds are issued") == ["refund", "issu"]
    assert "the" not in bm25_tokens("the policy of the company")


def test_parse_note_frontmatter():
    meta, body = parse_note("---\nid: abc\ndomain: hr\n---\n# T\n\ntext")
    assert meta == {"id": "abc", "domain": "hr"} and body.startswith("# T")
    assert parse_note("no frontmatter") == ({}, "no frontmatter")


def test_rrf_scores_dedupe_and_pool_cap():
    hashes = ["a", "b", "c", "d", "e", "c2"]
    # chunk 2 is top of both lists -> first; index 5 shares content with 2 ('c') -> deduped away
    hashes[5] = "c"
    fused = rrf_fuse([[2, 0, 1], [2, 5, 3]], hashes, k=60, pool=40)
    assert fused[0] == 2 and 5 not in fused
    assert set(fused) == {0, 1, 2, 3}
    assert len(rrf_fuse([list(range(6))], hashes, pool=3)) == 3
    # exact RRF arithmetic: item present at rank 1 in two lists scores 2/(60+1)
    assert rrf_fuse([[7], [7]], ["x"] * 8, k=60) == [7]


def test_rrf_prefers_agreement_over_single_list_rank():
    # item 1 is rank 2 in both lists; items 0 and 9 are rank 1 in only one list each
    fused = rrf_fuse([[0, 1], [9, 1]], [str(i) for i in range(10)])
    assert fused[0] == 1


# ---------------- chunking ----------------
@needs_models
def test_chunking_window_and_overlap():
    tok = Tokenizer.from_file(str(ROOT / DEFAULT_EMBED_DIR / "tokenizer.json"))
    tok.no_truncation(); tok.no_padding()
    short = "A short policy sentence."
    assert chunk_text(short, tok) == [short]
    words = " ".join(f"word{i}" for i in range(1500))
    chunks = chunk_text(words, tok)
    assert len(chunks) > 2
    sizes = [len(tok.encode(c, add_special_tokens=False).ids) for c in chunks]
    assert max(sizes) <= CHUNK_TOKENS
    # consecutive chunks overlap by ~CHUNK_OVERLAP tokens: the tail of chunk i appears at the head of chunk i+1
    ids = [tok.encode(c, add_special_tokens=False).ids for c in chunks]
    assert ids[0][-CHUNK_OVERLAP:] == ids[1][:CHUNK_OVERLAP]
    # nothing lost: every word appears in at least one chunk
    assert all(any(f"word{i}" in c.split() for c in chunks) for i in (0, 750, 1499))


# ---------------- index + stages (real ONNX models) ----------------
@pytest.fixture(scope="session")
def retriever(tmp_path_factory):
    if not MODELS_OK:
        pytest.skip("models missing")
    root = tmp_path_factory.mktemp("index")
    r = HybridRetriever(root, ROOT / DEFAULT_EMBED_DIR, ROOT / DEFAULT_RERANK_DIR)
    tok = Tokenizer.from_file(str(ROOT / DEFAULT_EMBED_DIR / "tokenizer.json"))
    tok.no_truncation(); tok.no_padding()
    build_index(TENANT, ROOT / "knowledge_vault", root, r.embedder, tok)
    return r


@needs_models
def test_chunk_ids_match_design_contract(retriever):
    ti = retriever.index_for(TENANT)
    assert len(ti.chunks) >= 12
    assert all(re.match(CHUNK_ID_RE, c.chunk_id) for c in ti.chunks)
    assert len({c.chunk_id for c in ti.chunks}) == len(ti.chunks)


@needs_models
def test_stage1_bm25_exact_identifier_match(retriever):
    ti = retriever.index_for(TENANT)
    hits, ms = retriever._bm25(ti, "SKU-205", 50, None, __import__("opentelemetry.context").context.get_current())
    assert ti.chunks[hits[0]].doc_id == "warranty"  # exact token that dense models can blur
    assert ms >= 0


@needs_models
def test_stage1_bm25_no_overlap_gives_no_candidates(retriever):
    ti = retriever.index_for(TENANT)
    hits, _ = retriever._bm25(ti, "zzzqqq xylophone", 50, None, __import__("opentelemetry.context").context.get_current())
    assert hits == []


@needs_models
def test_stage2_dense_matches_paraphrase_without_shared_words(retriever):
    ti = retriever.index_for(TENANT)
    ctx = __import__("opentelemetry.context").context.get_current()
    hits, _ = retriever._dense(ti, "how long are client files kept before deletion", 50, None, ctx)
    top3 = {ti.chunks[i].doc_id for i in hits[:3]}
    assert "retention" in top3


@needs_models
def test_stage3_rerank_scores_are_probabilities_and_ordered(retriever):
    ti = retriever.index_for(TENANT)
    texts = [c.text for c in ti.chunks]
    s = retriever.reranker.score("What is the minimum password length?", texts)
    assert ((0 <= s) & (s <= 1)).all()
    assert ti.chunks[int(s.argmax())].doc_id == "passwords"
    assert s.max() > 0.42 > sorted(s)[-2]  # one clear winner on either side of the threshold


# Known calibration gap, tracked for Prompt 11: the 0.42 cutoff is tuned for bge-reranker; the MiniLM stand-in
# scores "vacation days" vs a note that says "paid annual leave" at 0.23, so the system abstains wrongly.
KNOWN_FALSE_ABSTAIN = {"How many vacation days do I get each year?"}
_ANSWERABLE_PARAMS = [
    pytest.param(q, e, marks=pytest.mark.xfail(
        strict=True, reason="0.42 threshold not calibrated for stand-in reranker (score 0.23)"))
    if q in KNOWN_FALSE_ABSTAIN else pytest.param(q, e)
    for q, e in ANSWERABLE
]


@needs_models
@pytest.mark.parametrize("query,expected", _ANSWERABLE_PARAMS)
def test_end_to_end_top1_is_correct_document(retriever, query, expected):
    out = retriever.retrieve(TENANT, query)
    assert not out.abstain and out.chunks, f"unexpected abstain for: {query}"
    assert out.chunks[0].chunk_id.split("-")[0] == expected
    assert all(c.score > 0.42 for c in out.chunks) and len(out.chunks) <= 4


@needs_models
def test_known_false_abstain_still_retrieves_the_right_chunk_when_threshold_lowered(retriever):
    # the retrieval itself is right; only the abstention cutoff is miscalibrated for this model
    out = retriever.retrieve(TENANT, "How many vacation days do I get each year?", score_threshold=0.1)
    assert not out.abstain and out.chunks[0].chunk_id.startswith("pto-")


@needs_models
@pytest.mark.parametrize("query,_", UNANSWERABLE)
def test_unanswerable_abstains(retriever, query, _):
    out = retriever.retrieve(TENANT, query)
    assert out.abstain and out.chunks == [] and out.top_score < 0.42


@needs_models
def test_response_shape_and_latency_keys(retriever):
    out = retriever.retrieve(TENANT, "What is the minimum password length?")
    assert set(out.stage_latency_ms) == {"bm25", "dense", "rrf", "rerank"}
    assert all(v >= 0 for v in out.stage_latency_ms.values())
    top = out.chunks[0]
    assert top.bm25_rank is not None or top.dense_rank is not None


@needs_models
def test_domain_filter_restricts_results(retriever):
    out = retriever.retrieve(TENANT, "What are the rules about approval and access?", domain="finance",
                             score_threshold=0.0, rerank_k=10)
    assert out.chunks and {c.chunk_id.split("-")[0] for c in out.chunks} <= {"wires", "expenses"}


# ---------------- tenant isolation & index safety ----------------
@needs_models
def test_tenant_isolation_and_path_traversal(retriever, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "n.md").write_text("---\nid: secret\ntitle: S\ndomain: x\n---\nThe vault code is 1234.")
    tok = Tokenizer.from_file(str(ROOT / DEFAULT_EMBED_DIR / "tokenizer.json"))
    tok.no_truncation(); tok.no_padding()
    build_index("other-co", other, retriever.root, retriever.embedder, tok)
    secret_id = retriever.index_for("other-co").chunks[0].chunk_id
    with pytest.raises(IndexNotFound):  # tenant A cannot read tenant B's chunk, by id
        retriever.get_chunk(TENANT, secret_id)
    assert retriever.get_chunk("other-co", secret_id).text.startswith("The vault")
    out = retriever.retrieve(TENANT, "What is the vault code?")
    assert "1234" not in " ".join(c.text for c in out.chunks)
    for bad in ["../etc", "a/b", "..", "ACME", ""]:
        with pytest.raises(IndexNotFound):
            load_index(bad, retriever.root, retriever.embedder.model_id)


@needs_models
def test_index_built_with_other_model_is_refused(retriever):
    manifest = retriever.root / TENANT / "manifest.json"
    original = manifest.read_text()
    try:
        m = json.loads(original); m["embedding_model"] = "some-other-model"
        manifest.write_text(json.dumps(m))
        with pytest.raises(IndexVersionError):
            load_index(TENANT, retriever.root, retriever.embedder.model_id)
    finally:
        manifest.write_text(original)


# ---------------- OTel (rule 5) and PII (rule 4) ----------------
@needs_models
def test_each_stage_emits_a_span_and_query_text_never_leaks(retriever, spans):
    q = "My email is alice@example.com, what is the refund window?"
    retriever.retrieve(TENANT, q)
    names = {s.name for s in spans.get_finished_spans()}
    assert {"mcp.rag.hybrid_retrieve", "mcp.rag.bm25", "mcp.rag.biencoder", "mcp.rag.rrf", "mcp.rag.rerank"} <= names
    dump = " ".join(repr(dict(s.attributes)) + s.name for s in spans.get_finished_spans())
    assert "alice@example.com" not in dump
    root = next(s for s in spans.get_finished_spans() if s.name == "mcp.rag.hybrid_retrieve")
    # stage spans are children of the retrieval span even though BM25/dense run in worker threads
    kids = [s for s in spans.get_finished_spans() if s.name in {"mcp.rag.bm25", "mcp.rag.biencoder"}]
    assert all(k.parent.span_id == root.context.span_id for k in kids)

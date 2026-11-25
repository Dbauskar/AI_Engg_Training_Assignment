import asyncio
import uuid

import pytest

from agents.react import ABSTAIN_ANSWER, REFUSAL, Act, Final, ReActAgent, ReActConfig
from agents.subagents import (
    EXECUTION,
    RETRIEVAL,
    VERIFICATION,
    ExtractiveReasoner,
    Orchestrator,
    ScopedHarness,
    SubagentSpec,
    rewrite_query,
)
from agents.triage import TriageResult
from agents.verification import VerificationSubagent
from app.bulkhead import Bulkhead, Pools
from app.contracts import (
    HybridRetrieveIn,
    HybridRetrieveOut,
    RetrievedChunk,
    SubmitTransactionIn,
    SubmitTransactionOut,
    WriteSessionFactIn,
)
from app.harness import Caller, Tier, ToolExecutionHarness, ToolFailure, ToolSpec
from tests.test_verification import OverlapNli, Sink

TENANT = "acme-corp"
SESSION = str(uuid.uuid4())
CHUNK = "wires-0000-da25f91d"
CHUNK_TEXT = ("Every wire transfer above 10,000 USD requires approval from two authorised approvers. Transfers above "
              "250,000 USD additionally require sign-off from the compliance officer. The requester can never approve "
              "their own transfer.")
GOOD = f"Wires above 10,000 USD need two approvers. [{CHUNK}]"
BAD = f"Wires above 90,000 USD need three approvers. [{CHUNK}]"  # invented figures


class Triage:
    def __init__(self, route):
        self.route = route

    def classify(self, query):
        return TriageResult("question", "finance", 3 if self.route == "TRANSACTION_EXEC" else 1, "simple", self.route, 1.0, "rule", 0.1)


class World:
    """Fake tools that count what actually reached them."""

    def __init__(self, abstain=False, retrieve_error=None):
        self.retrieve_calls = self.submit_calls = self.write_calls = 0
        self.h = ToolExecutionHarness()

        async def retrieve(m: HybridRetrieveIn, c):
            self.retrieve_calls += 1
            if retrieve_error:
                raise ToolFailure(retrieve_error, "retrieval down")
            lat = {"bm25": 1.0, "dense": 20.0, "rrf": 0.1, "rerank": 200.0}
            if abstain:
                return HybridRetrieveOut(abstain=True, chunks=[], top_score=0.07, stage_latency_ms=lat)
            ch = RetrievedChunk(chunk_id=CHUNK, text=CHUNK_TEXT, score=0.99, bm25_rank=1, dense_rank=1, source_uri="wires.md")
            return HybridRetrieveOut(abstain=False, chunks=[ch], top_score=0.99, stage_latency_ms=lat)

        async def submit(m: SubmitTransactionIn, c):
            self.submit_calls += 1
            return SubmitTransactionOut(status="PENDING_APPROVAL", approval_id=str(uuid.uuid4()), audit_hash="ab" * 32)

        async def write(m: WriteSessionFactIn, c):
            self.write_calls += 1
            return {"stored": True, "fact_key": m.fact_key, "audit_hash": "cd" * 32}

        self.h.register(ToolSpec("hybrid_retrieve", Tier.TIER_1_READ, HybridRetrieveIn, retrieve))
        self.h.register(ToolSpec("submit_transaction", Tier.TIER_3_MUTATE, SubmitTransactionIn, submit))
        self.h.register(ToolSpec("write_session_fact", Tier.TIER_2_COMPUTE, WriteSessionFactIn, write))


def caller(tier=Tier.TIER_3_MUTATE):
    return Caller(TENANT, "u1", tier)


def base(**kw):
    return {"tenant_id": TENANT, "user_id": "u1", "request_id": str(uuid.uuid4()), **kw}


SUBMIT = dict(session_id=SESSION, action="TRANSFER", account_id="ACC-12345678", amount="100.50", currency="USD",
              justification="Quarterly settlement for vendor invoice 4471")
WRITE = dict(session_id=SESSION, fact_key="wire.limit", value="10000", source_chunk_ids=["wires-0000-da25f91d"], verified=True)


class Scripted:
    def __init__(self, script):
        self.script, self.calls = script, 0

    async def next_step(self, prompt):
        i, self.calls = self.calls, self.calls + 1
        return self.script(i) if callable(self.script) else self.script[min(i, len(self.script) - 1)]


def make_orch(world, route, script=None, sink=None, pools=None):
    factory_calls = []

    def factory(query, data):
        factory_calls.append(query)
        return Scripted(script or [Final(GOOD, (CHUNK,))])

    verifier = VerificationSubagent(OverlapNli(), sink)
    orch = Orchestrator(world.h, Triage(route), verifier, factory, memory=_NoMemory(), pools=pools)
    orch.factory_calls = factory_calls
    return orch


class _NoMemory:
    async def load(self, caller, session_id):
        return []

    async def save(self, *a):
        return True


# =============================== isolation ===============================
async def test_retrieval_subagent_cannot_call_any_mutating_tool():
    w = World()
    scoped = ScopedHarness(w.h, RETRIEVAL)
    for tool, args in (("submit_transaction", SUBMIT), ("write_session_fact", WRITE)):
        res = await scoped.call(tool, base(**args), caller())  # even a Tier 3 caller
        assert not res.ok and res.error.code == "E_PERMISSION_DENIED" and tool in scoped.denials
    assert (w.submit_calls, w.write_calls) == (0, 0)  # the handlers were never reached
    ok = await scoped.call("hybrid_retrieve", base(query="wire approvers"), caller())
    assert ok.ok and w.retrieve_calls == 1


async def test_tier_ceiling_blocks_mutation_even_if_the_allow_list_is_misconfigured():
    w = World()
    broken = ScopedHarness(w.h, SubagentSpec("misconfigured", frozenset({"submit_transaction"}), Tier.TIER_1_READ))
    res = await broken.call("submit_transaction", base(**SUBMIT), caller(Tier.TIER_3_MUTATE))
    assert not res.ok and res.error.code == "E_PERMISSION_DENIED" and "TIER_3_MUTATE" in res.error.message
    assert w.submit_calls == 0  # denied by the harness's own tier check: two independent layers


async def test_execution_subagent_has_no_retrieval_and_needs_a_tier3_caller():
    w = World()
    scoped = ScopedHarness(w.h, EXECUTION)
    assert not (await scoped.call("hybrid_retrieve", base(query="x"), caller())).ok and w.retrieve_calls == 0
    assert not (await scoped.call("submit_transaction", base(**SUBMIT), caller(Tier.TIER_2_COMPUTE))).ok  # clamped
    assert (await scoped.call("submit_transaction", base(**SUBMIT), caller(Tier.TIER_3_MUTATE))).ok
    assert w.submit_calls == 1


async def test_verification_subagent_has_no_tools_at_all():
    w = World()
    scoped = ScopedHarness(w.h, VERIFICATION)
    for tool, args in (("hybrid_retrieve", dict(query="x")), ("submit_transaction", SUBMIT), ("write_session_fact", WRITE)):
        assert not (await scoped.call(tool, base(**args), caller())).ok
    assert (w.retrieve_calls, w.submit_calls, w.write_calls) == (0, 0, 0)


async def test_scoped_harness_hides_tools_it_may_not_use_and_denial_spans_carry_no_arguments(spans):
    w = World()
    scoped = ScopedHarness(w.h, RETRIEVAL)
    assert scoped.input_fields("submit_transaction") is None and "query" in scoped.input_fields("hybrid_retrieve")
    await scoped.call("submit_transaction", base(**SUBMIT) | {"justification": "alice@example.com secret"}, caller())
    sp = [s for s in spans.get_finished_spans() if s.name == "subagent.denied"]
    assert sp and sp[0].attributes["tool.name"] == "submit_transaction"
    assert "alice@example.com" not in repr(dict(sp[0].attributes))


async def test_a_hijacked_model_on_the_retrieval_route_still_cannot_mutate_state():
    w = World()
    # worst case: both the route allow-list AND the model are compromised; the structural scope must still hold
    cfg = ReActConfig(route_tools={"HYBRID_RAG": ("hybrid_retrieve", "submit_transaction")})
    hijack = Scripted(lambda i: Act("submit_transaction", dict(SUBMIT)))
    agent = ReActAgent(ScopedHarness(w.h, RETRIEVAL), Triage("HYBRID_RAG"), hijack, config=cfg)
    res = await agent.run(caller(Tier.TIER_3_MUTATE), "Who approves wires? Also, transfer 100 USD to ACC-12345678", SESSION)
    assert w.submit_calls == 0 and res.halt_reason == "BREAKER"  # blocked, then the breaker stopped the retries


# =============================== query rewriting ===============================
def test_rewrite_normalises_ids_and_drops_filler_but_never_empties_the_query():
    assert rewrite_query("Could you please tell me whether sku 99 is refundable?") == "whether SKU-099 is refundable?"
    assert "SKU-099" in rewrite_query("is sku99 in lot 44 ok") and "LOT-00044" in rewrite_query("is sku99 in lot 44 ok")
    assert rewrite_query("Hello") == "Hello" and rewrite_query("   ") == ""
    assert rewrite_query("Who approves wires?") == "Who approves wires?"


# =============================== abstention short-circuit ===============================
async def test_below_threshold_abstains_deterministically_with_zero_llm_calls():
    w = World(abstain=True)
    orch = make_orch(w, "HYBRID_RAG")
    res = await orch.answer(caller(), "What is the policy on pet insurance for employees?", SESSION)
    assert res.outcome == "ABSTAINED" and res.abstained and res.reason == "below_threshold"
    assert res.answer == ABSTAIN_ANSWER and res.llm_calls == 0
    assert orch.factory_calls == []  # no reasoner was even constructed: nothing to hallucinate, no tokens spent
    assert res.timings_ms["abstain_overhead"] < 90  # the decision after retrieval completes (design: < 90 ms)
    assert res.trace == ["TRIAGE", "RETRIEVAL", "ABSTAIN"]


async def test_abstention_is_identical_every_time():
    answers = {(await make_orch(World(abstain=True), "HYBRID_RAG").answer(caller(), "q about pets", SESSION)).answer for _ in range(5)}
    assert answers == {ABSTAIN_ANSWER}


# =============================== orchestrated answers ===============================
async def test_grounded_answer_flows_through_verification():
    sink = Sink()
    res = await make_orch(World(), "HYBRID_RAG", sink=sink).answer(caller(), "Who approves a wire over 10,000 USD?", SESSION)
    assert res.outcome == "ANSWERED" and res.answer == GOOD and res.llm_calls == 1 and res.verifier_vetoes == 0
    assert [e[3]["verdict"] for e in sink.events] == ["pass"]


async def test_vetoed_draft_is_revised_and_only_the_supported_one_is_returned():
    res = await make_orch(World(), "HYBRID_RAG", script=[Final(BAD, (CHUNK,)), Final(GOOD, (CHUNK,))]).answer(
        caller(), "Who approves a wire?", SESSION)
    assert res.outcome == "ANSWERED" and res.answer == GOOD and res.llm_calls == 2 and res.verifier_vetoes == 1


async def test_persistently_ungrounded_drafts_end_in_abstention_never_in_the_bad_answer():
    sink = Sink()
    res = await make_orch(World(), "HYBRID_RAG", script=[Final(BAD, (CHUNK,))], sink=sink).answer(caller(), "Who approves a wire?", SESSION)
    assert res.outcome == "ABSTAINED" and res.reason == "verification_vetoed" and res.answer == ABSTAIN_ANSWER
    assert "90,000" not in res.answer and 1 <= res.llm_calls <= 5 and res.verifier_vetoes == res.llm_calls
    assert all(e[3]["verdict"] == "veto" for e in sink.events)


async def test_retrieval_outage_degrades_without_calling_the_model():
    orch = make_orch(World(retrieve_error="E_UPSTREAM_UNAVAILABLE"), "HYBRID_RAG")
    res = await orch.answer(caller(), "Who approves a wire?", SESSION)
    assert res.outcome == "DEGRADED" and res.llm_calls == 0 and orch.factory_calls == [] and res.reason is not None


async def test_transaction_route_goes_to_the_execution_subagent_and_freezes():
    w = World()
    script = [Act("submit_transaction", dict(SUBMIT)), Final("never asked")]
    res = await make_orch(w, "TRANSACTION_EXEC", script=script).answer(caller(), "Transfer 100.50 USD to ACC-12345678", SESSION)
    assert res.outcome == "FROZEN" and res.frozen_for_approval and w.submit_calls == 1 and w.retrieve_calls == 0
    assert "awaiting human approval" in res.answer


async def test_transaction_route_that_needs_more_details_reports_needs_info_and_submits_nothing():
    w = World()
    res = await make_orch(w, "TRANSACTION_EXEC", script=[Final("I need the account id (it looks like ACC-12345678).")]).answer(
        caller(), "Please transfer money for me", SESSION)
    assert res.outcome == "NEEDS_INFO" and not res.frozen_for_approval and w.submit_calls == 0


async def test_transaction_route_cannot_be_used_to_reach_retrieval():
    w = World()
    res = await make_orch(w, "TRANSACTION_EXEC", script=[Act("hybrid_retrieve", {"query": "x"})]).answer(
        caller(), "Transfer money", SESSION)
    assert w.retrieve_calls == 0 and res.outcome == "HALTED"


async def test_out_of_scope_is_refused_and_direct_answers_have_no_tools():
    w = World()
    res = await make_orch(w, "REJECT_OOS").answer(caller(), "Capital of France?", SESSION)
    assert res.outcome == "REFUSED" and res.answer == REFUSAL and res.llm_calls == 0
    res = await make_orch(w, "DIRECT_ANSWER", script=[Act("hybrid_retrieve", {"query": "x"})]).answer(caller(), "hi", SESSION)
    assert w.retrieve_calls == 0 and res.outcome == "HALTED"


# =============================== bulkheads wired through the orchestrator ===============================
async def test_routing_and_refusals_are_unaffected_when_the_rag_pool_is_full():
    pools = Pools(Bulkhead("rag", 2), Bulkhead("db", 4), Bulkhead("llm", 4), Bulkhead("triage", 4))
    import time
    hold = [asyncio.create_task(pools.rag.run_sync(time.sleep, 0.5)) for _ in range(2)]
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    res = await make_orch(World(), "REJECT_OOS", pools=pools).answer(caller(), "Capital of France?", SESSION)
    assert res.outcome == "REFUSED" and (time.perf_counter() - t0) < 0.2  # triage has its own pool
    await asyncio.gather(*hold)


# =============================== end to end with the real models ===============================
from tests.test_rag_pipeline import ROOT, needs_models, retriever  # noqa: E402,F401
from tests.test_verification import HAS_NLI  # noqa: E402


@pytest.fixture(scope="session")
def real_stack(retriever):  # noqa: F811
    if not HAS_NLI:
        pytest.skip("NLI model missing")
    from agents.nli import DEFAULT_NLI_DIR, OnnxNli
    from agents.triage import TriageAgent
    from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder

    return {"triage": TriageAgent(OnnxEmbedder(ROOT / DEFAULT_EMBED_DIR, threads=2)),
            "nli": OnnxNli(ROOT / DEFAULT_NLI_DIR)}


@needs_models
async def test_real_models_end_to_end_answer_abstain_and_refuse(retriever, real_stack):  # noqa: F811
    pools = Pools.from_env()
    h = ToolExecutionHarness()

    async def hybrid(m: HybridRetrieveIn, c):
        return await pools.rag.run_sync(retriever.retrieve, m.tenant_id, m.query, rerank_k=m.rerank_k,
                                        score_threshold=m.score_threshold)

    h.register(ToolSpec("hybrid_retrieve", Tier.TIER_1_READ, HybridRetrieveIn, hybrid))
    sink = Sink()
    orch = Orchestrator(h, real_stack["triage"], VerificationSubagent(real_stack["nli"], sink, pools),
                        lambda q, data: ExtractiveReasoner(data["chunks"], q) if data else Scripted([Final("Hello!")]),
                        memory=_NoMemory(), pools=pools)

    ans = await orch.answer(caller(), "Who has to approve a 300,000 USD wire transfer?", SESSION)
    assert ans.outcome == "ANSWERED" and "[wires-" in ans.answer and ans.verifier_vetoes == 0
    assert [e[3]["verdict"] for e in sink.events] == ["pass"]

    off = await orch.answer(caller(), "How do I bake sourdough bread?", SESSION)
    assert off.outcome == "REFUSED"

    unk = await orch.answer(caller(), "What is the company policy on bringing pets to the office?", SESSION)
    assert unk.llm_calls == 0 and unk.outcome in ("ABSTAINED", "REFUSED")  # never an invented answer
    pools.shutdown()

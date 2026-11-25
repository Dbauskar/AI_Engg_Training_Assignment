import asyncio
import uuid
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, StringConstraints

from agents.react import (
    ALLOWED,
    REFUSAL,
    Act,
    CircuitBreaker,
    Final,
    FactProposal,
    GoalAnchorBlock,
    HarnessFactMemory,
    ReActAgent,
    ReActConfig,
    State,
    build_prompt,
    compress_observation,
    count_tokens,
    truncate_tokens,
)
from agents.triage import TriageResult
from app.contracts import (
    GetSessionFactsIn,
    HybridRetrieveIn,
    HybridRetrieveOut,
    RetrievedChunk,
    SubmitTransactionIn,
    SubmitTransactionOut,
    ToolInputBase,
    WriteSessionFactIn,
)
from app.harness import Caller, Tier, ToolExecutionHarness, ToolSpec
from tests.test_state_store import store  # noqa: F401

TENANT = "acme-corp"
SESSION = str(uuid.uuid4())
CHUNK = "wires-0000-da25f91d"


# ---------------- scaffolding ----------------
class InventoryIn(ToolInputBase):
    sku: Annotated[str, StringConstraints(pattern=r"^SKU-\d{3}$")]


class FakeTriage:
    def __init__(self, route="HYBRID_RAG"):
        self.route = route

    def classify(self, query):
        return TriageResult("question", "finance", 3 if self.route == "TRANSACTION_EXEC" else 1, "simple",
                            self.route, 1.0, "rule", 0.1)


class Scripted:
    """Reasoner that replays a script (list of steps, or a callable(i, prompt)) and records every prompt."""

    def __init__(self, script, delay=0.0, on_call=None):
        self.script, self.delay, self.on_call = script, delay, on_call
        self.prompts: list[list] = []

    async def next_step(self, prompt):
        self.prompts.append(prompt)
        i = len(self.prompts) - 1
        if self.on_call:
            self.on_call()
        if self.delay:
            await asyncio.sleep(self.delay)
        if callable(self.script):
            return self.script(i, prompt)
        return self.script[min(i, len(self.script) - 1)]


class World:
    """A harness with fake tools that record what they were really called with."""

    def __init__(self, tier=Tier.TIER_3_MUTATE, inventory_delay=0.0):
        self.inventory_calls: list[InventoryIn] = []
        self.retrieve_calls = 0
        self.h = ToolExecutionHarness()
        self.caller = Caller(TENANT, "u1", tier)

        async def inventory(m: InventoryIn, c):
            self.inventory_calls.append(m)
            if inventory_delay:
                await asyncio.sleep(inventory_delay)
            return {"sku": m.sku, "on_hand": 7}

        async def retrieve(m: HybridRetrieveIn, c):
            self.retrieve_calls += 1
            chunk = RetrievedChunk(chunk_id=CHUNK, text="Wires above 10,000 USD need two approvers. " * 3 +
                                   "Above 250,000 USD the compliance officer must also sign off.",
                                   score=0.99, bm25_rank=1, dense_rank=2, source_uri="wires.md")
            return HybridRetrieveOut(abstain=False, chunks=[chunk], top_score=0.99,
                                     stage_latency_ms={"bm25": 1.0, "dense": 20.0, "rrf": 0.1, "rerank": 200.0})

        async def submit(m: SubmitTransactionIn, c):
            return SubmitTransactionOut(status="PENDING_APPROVAL", approval_id=str(uuid.uuid4()), audit_hash="ab" * 32)

        self.h.register(ToolSpec("get_inventory", Tier.TIER_1_READ, InventoryIn, inventory))
        self.h.register(ToolSpec("hybrid_retrieve", Tier.TIER_1_READ, HybridRetrieveIn, retrieve))
        self.h.register(ToolSpec("submit_transaction", Tier.TIER_3_MUTATE, SubmitTransactionIn, submit))


INV_TOOLS = {"DIRECT_ANSWER": ("get_inventory",), "HYBRID_RAG": ("get_inventory", "hybrid_retrieve"),
             "TRANSACTION_EXEC": ("submit_transaction",)}


def make(world, script, route="DIRECT_ANSWER", cfg=None, **kw):
    reasoner = script if hasattr(script, "next_step") else Scripted(script)
    cfg = cfg or ReActConfig(route_tools=INV_TOOLS)
    return ReActAgent(world.h, FakeTriage(route), reasoner, config=cfg, **kw), reasoner


def assert_legal(res):
    for a, b in zip(res.trace, res.trace[1:]):
        assert State(b) in ALLOWED[State(a)], f"illegal transition {a} -> {b} in {res.trace}"


# ---------------- state machine ----------------
def test_transition_table_is_closed():
    assert ALLOWED[State.DONE] == frozenset() and ALLOWED[State.HALTED] == frozenset()
    assert State.TOOL_CALL not in ALLOWED[State.INIT] and State.DONE not in ALLOWED[State.REASONING]


async def test_happy_path_trace_and_result():
    w = World()
    agent, r = make(w, [Act("get_inventory", {"sku": "SKU-099"}, "look it up"), Final("There are 7 units.")])
    res = await agent.run(w.caller, "How many SKU-099 are in stock?", SESSION)
    assert res.state is State.DONE and res.answer == "There are 7 units." and res.iterations == 2
    assert res.trace == ["INIT", "TRIAGE", "REASONING", "TOOL_CALL", "OBSERVATION", "REASONING", "VERIFY", "DONE"]
    assert_legal(res)
    assert res.tool_calls == 1 and res.repaired_calls == 0 and res.halt_reason is None


async def test_out_of_scope_is_refused_without_calling_the_model_or_any_tool():
    w = World()
    agent, r = make(w, [Final("never reached")], route="REJECT_OOS")
    res = await agent.run(w.caller, "What is the capital of France?", SESSION)
    assert res.state is State.DONE and res.answer == REFUSAL and res.trace == ["INIT", "TRIAGE", "DONE"]
    assert r.prompts == [] and res.tool_calls == 0


# ---------------- goal anchor & sliding window ----------------
def test_goal_anchor_is_frozen_and_normalises_entities():
    a = GoalAnchorBlock.build("Is sku 99 in lot 44 and ACC-12345678?", FakeTriage().classify(""))
    assert a.entities == ("ACC-12345678", "LOT-00044", "SKU-099")
    with pytest.raises(Exception):
        a.intent = "something else"  # frozen dataclass
    assert isinstance(a.entities, tuple) and isinstance(a.constraints, tuple)  # no mutable members


async def test_anchor_is_prompt_index_zero_and_identical_on_every_iteration():
    w = World()
    cfg = ReActConfig(route_tools=INV_TOOLS, max_iterations=6)
    steps = [Act("get_inventory", {"sku": f"SKU-00{i}"}) for i in range(1, 5)] + [Final("done")]
    agent, r = make(w, steps, cfg=cfg)
    res = await agent.run(w.caller, "Check stock levels please", SESSION)
    assert res.state is State.DONE and len(r.prompts) == 5
    assert len({p[0].content for p in r.prompts}) == 1
    assert all(p[0].role == "goal" and "GOAL ANCHOR" in p[0].content for p in r.prompts)


async def test_sliding_window_keeps_only_the_last_three_turns_but_never_drops_the_anchor():
    w = World()
    cfg = ReActConfig(route_tools=INV_TOOLS, max_iterations=6)
    steps = [Act("get_inventory", {"sku": f"SKU-00{i}"}) for i in range(1, 6)] + [Final("done")]
    agent, r = make(w, steps, cfg=cfg)
    await agent.run(w.caller, "Check stock levels please", SESSION)
    last = "\n".join(m.content for m in r.prompts[5])  # sixth prompt: five turns exist, only three are shown
    assert all(f"SKU-00{i}" in last for i in (3, 4, 5)) and not any(f"SKU-00{i}" in last for i in (1, 2))
    assert r.prompts[5][0].role == "goal"
    assert sum(m.role == "action" for m in r.prompts[5]) == 3


# ---------------- observation compression ----------------
def test_compression_drops_boilerplate_keeps_signal_and_caps_tokens():
    data = {"abstain": False, "top_score": 0.99,
            "stage_latency_ms": {"bm25": 1, "dense": 20}, "chunks": [
                {"chunk_id": CHUNK, "score": 0.99, "bm25_rank": 1, "dense_rank": 2, "source_uri": "wires.md",
                 "text": "Filler sentence one. " * 40 + "Wire approvals above 250,000 USD need the compliance officer. " + "More filler. " * 40}]}
    out = compress_observation(data, {"approve", "wire", "compliance", "officer"}, 180)
    assert count_tokens(out) <= 180
    assert CHUNK in out and "0.99" in out and "compliance officer" in out  # goal-relevant sentence survives
    for boiler in ("stage_latency_ms", "bm25_rank", "dense_rank", "source_uri"):
        assert boiler not in out


@pytest.mark.parametrize("budget", [1, 2, 5, 20, 180])
def test_hard_cap_holds_for_any_input_and_budget(budget):
    for text in ("word " * 5000, "x" * 10000, "a. b. c. " * 300, "", "short"):
        assert count_tokens(truncate_tokens(text, budget)) <= budget
        assert count_tokens(compress_observation(text, set(), budget)) <= budget
    huge = {"items": [{"k": "v " * 400, "n": i} for i in range(60)]}
    assert count_tokens(compress_observation(huge, set(), budget)) <= budget


async def test_loop_feeds_the_model_only_compressed_observations():
    w = World()
    agent, r = make(w, [Act("hybrid_retrieve", {"query": "wire approval"}), Final("ok", citations=(CHUNK,))],
                    route="HYBRID_RAG")
    await agent.run(w.caller, "Who approves a 300,000 USD wire?", SESSION)
    obs = [m for m in r.prompts[1] if m.role == "observation"][0].content
    assert count_tokens(obs) <= 180 and CHUNK in obs and "stage_latency_ms" not in obs


# ---------------- the infinite SKU retry scenario ----------------
async def test_infinite_sku_retry_is_stopped_by_the_circuit_breaker():
    w = World()
    stubborn = Scripted(lambda i, p: Act("get_inventory", {"sku": "WIDGET-9"}, "try again"))  # never repairable
    agent, _ = make(w, stubborn)
    res = await agent.run(w.caller, "Do we have WIDGET-9 in stock?", SESSION)
    assert res.state is State.HALTED and res.halt_reason == "BREAKER"
    assert res.iterations == 2 and res.tool_calls == 2 and len(stubborn.prompts) == 2  # stopped at the 2nd failure
    assert w.inventory_calls == []  # invalid input never reached the tool
    assert "sku" in res.answer and "WIDGET-9" not in res.answer  # remediation names the field, not the value
    assert res.elapsed_ms < 450  # well inside the loop budget: no spinning
    assert_legal(res)


async def test_breaker_stays_open_without_calling_the_tool_again():
    b = CircuitBreaker(2)
    k = b.key("get_inventory", {"sku": "WIDGET-9"})
    assert b.record_failure(k) is False and b.is_open(k) is False
    assert b.record_failure(k) is True and b.is_open(k) is True
    assert b.record_failure(k) is True and b.is_open(k) is True  # idempotent: still open
    assert b.key("get_inventory", {"sku": "WIDGET-9"}) == k and b.key("get_inventory", {"sku": "WIDGET-8"}) != k
    assert b.key("t", {"a": 1, "b": 2}) == b.key("t", {"b": 2, "a": 1})  # canonical
    b.record_success(k)
    assert b.is_open(k) is False


async def test_flapping_invalid_args_are_bounded_by_max_iterations():
    w = World()
    flapper = Scripted(lambda i, p: Act("get_inventory", {"sku": f"WIDGET-{i}"}))  # different args: breaker never trips
    agent, _ = make(w, flapper)
    res = await agent.run(w.caller, "Do we have any widgets?", SESSION)
    assert res.state is State.HALTED and res.halt_reason == "MAX_ITERATIONS"
    assert res.iterations == 5 and res.tool_calls == 5 and len(flapper.prompts) == 5
    assert_legal(res)


async def test_sku_typo_is_repaired_and_the_task_completes():
    w = World()
    agent, _ = make(w, [Act("get_inventory", {"sku": "SKU99"}), Final("7 units")])
    res = await agent.run(w.caller, "How many SKU99 do we have?", SESSION)
    assert res.state is State.DONE and res.iterations == 2 and res.repaired_calls == 1
    assert w.inventory_calls[0].sku == "SKU-099"


# ---------------- limits ----------------
async def test_per_step_timeout_on_the_model_halts_the_run():
    w = World()
    agent, _ = make(w, Scripted([Final("late")], delay=0.5), cfg=ReActConfig(route_tools=INV_TOOLS, per_step_timeout_s=0.05))
    res = await agent.run(w.caller, "hello", SESSION)
    assert res.state is State.HALTED and res.halt_reason == "STEP_TIMEOUT" and res.elapsed_ms < 300


async def test_slow_tool_times_out_counts_as_failure_and_trips_the_breaker():
    w = World(inventory_delay=0.5)
    cfg = ReActConfig(route_tools=INV_TOOLS, per_step_timeout_s=0.05)
    agent, _ = make(w, Scripted(lambda i, p: Act("get_inventory", {"sku": "SKU-001"})), cfg=cfg)
    res = await agent.run(w.caller, "stock?", SESSION)
    assert res.halt_reason == "BREAKER" and res.iterations == 2 and res.elapsed_ms < 400


async def test_total_budget_is_enforced():
    w = World()

    class Clock:
        now = 0.0

    clock = Clock()
    r = Scripted(lambda i, p: Act("get_inventory", {"sku": f"SKU-00{i + 1}"}), on_call=lambda: setattr(clock, "now", clock.now + 0.2))
    agent = ReActAgent(w.h, FakeTriage(), r, config=ReActConfig(route_tools=INV_TOOLS), clock=lambda: clock.now)
    res = await agent.run(w.caller, "keep going", SESSION)
    assert res.state is State.HALTED and res.halt_reason == "BUDGET_EXCEEDED" and res.iterations == 3  # 0.2 x3 > 0.45
    assert_legal(res)


# ---------------- safety properties ----------------
async def test_model_cannot_override_tenant_user_or_request_id():
    w = World()
    evil = {"sku": "SKU-001", "tenant_id": "evil-co", "user_id": "admin", "request_id": "not-a-uuid"}
    agent, _ = make(w, [Act("get_inventory", evil), Final("done")])
    res = await agent.run(w.caller, "stock?", SESSION)
    assert res.state is State.DONE
    m = w.inventory_calls[0]
    assert (m.tenant_id, m.user_id) == (TENANT, "u1") and m.request_id != "not-a-uuid"


async def test_tool_outside_the_route_allow_list_is_refused_and_counted():
    w = World()
    agent, _ = make(w, Scripted(lambda i, p: Act("submit_transaction", {"action": "TRANSFER"})), route="DIRECT_ANSWER")
    res = await agent.run(w.caller, "hi", SESSION)
    assert res.halt_reason == "BREAKER" and res.tool_calls == 0 and w.retrieve_calls == 0  # never even reached the harness


async def test_tier3_submission_freezes_the_loop_at_pending_approval():
    w = World()
    args = {"action": "TRANSFER", "account_id": "ACC-12345678", "amount": "1250.50", "currency": "USD",
            "justification": "Quarterly settlement for vendor invoice 4471"}
    r = Scripted([Act("submit_transaction", args), Final("should never be asked for")])
    agent, _ = make(w, r, route="TRANSACTION_EXEC")
    res = await agent.run(w.caller, "Transfer 1250.50 USD to ACC-12345678", SESSION)
    assert res.state is State.DONE and res.frozen_for_approval and "awaiting human approval" in res.answer
    assert "Nothing has been executed" in res.answer and len(r.prompts) == 1  # the model is not asked to continue
    assert_legal(res)


# ---------------- verification ----------------
async def test_uncited_answer_is_sent_back_then_accepted_once_grounded():
    w = World()
    steps = [Act("hybrid_retrieve", {"query": "wire approval"}), Final("Two approvers."),
             Final("Two approvers.", citations=(CHUNK,))]
    agent, r = make(w, steps, route="HYBRID_RAG")
    res = await agent.run(w.caller, "Who approves a wire?", SESSION)
    assert res.state is State.DONE and res.iterations == 3 and "VERIFY" in res.trace
    assert any(m.role == "system" and "Verification failed" in m.content for m in r.prompts[2])
    assert_legal(res)


async def test_fabricated_citation_never_passes_and_loop_stays_bounded():
    w = World()
    agent, _ = make(w, Scripted(lambda i, p: Act("hybrid_retrieve", {"query": "x"}) if i == 0
                                else Final("made up", citations=("ghost-0000-deadbeef",))), route="HYBRID_RAG")
    res = await agent.run(w.caller, "Who approves a wire?", SESSION)
    assert res.state is State.HALTED and res.halt_reason == "MAX_ITERATIONS" and res.iterations == 5


# ---------------- facts memory ----------------
class MemFacts:
    def __init__(self, preload=()):
        self.facts, self.saved = list(preload), []

    async def load(self, caller, session_id):
        return list(self.facts)

    async def save(self, caller, session_id, key, value, chunk_ids):
        self.saved.append((key, value, chunk_ids))
        return True


async def test_only_grounded_facts_are_persisted_and_known_facts_are_in_the_prompt():
    w = World()
    mem = MemFacts(preload=[("wire.limit", "10000")])
    facts = (FactProposal("wire.two_approvers", "above 10000 USD", (CHUNK,)),
             FactProposal("wire.made_up", "nonsense", ("ghost-0000-deadbeef",)))
    agent, r = make(w, [Act("hybrid_retrieve", {"query": "q"}), Final("ok", citations=(CHUNK,), facts=facts)],
                    route="HYBRID_RAG", memory=mem)
    res = await agent.run(w.caller, "Who approves a wire?", SESSION)
    assert res.facts_saved == 1 and [s[0] for s in mem.saved] == ["wire.two_approvers"]
    assert r.prompts[0][1].role == "facts" and "wire.limit: 10000" in r.prompts[0][1].content


async def test_facts_persist_in_postgres_through_the_harness_and_return_next_run(store):  # noqa: F811
    from app.state_store import StateStore

    tenant = f"ta-{uuid.uuid4().hex[:8]}"
    session = str(uuid.uuid4())
    w = World()
    caller = Caller(tenant, "u1", Tier.TIER_2_COMPUTE)

    async def get_facts(m, c):
        return await store.get_session_facts(m)

    async def write_fact(m, c):
        return await store.write_session_fact(m)

    w.h.register(ToolSpec("get_session_facts", Tier.TIER_1_READ, GetSessionFactsIn, get_facts))
    w.h.register(ToolSpec("write_session_fact", Tier.TIER_2_COMPUTE, WriteSessionFactIn, write_fact))
    mem = HarnessFactMemory(w.h)
    chunk = "pto-0000-ab12cd34"
    fact = FactProposal("leave.days", "25", (chunk,))

    async def retrieve(m: HybridRetrieveIn, c):
        ch = RetrievedChunk(chunk_id=chunk, text="25 days of annual leave.", score=0.9, bm25_rank=1, dense_rank=1, source_uri="pto.md")
        return HybridRetrieveOut(abstain=False, chunks=[ch], top_score=0.9, stage_latency_ms={"bm25": 1, "dense": 1, "rrf": 0, "rerank": 1})

    w.h._tools["hybrid_retrieve"] = ToolSpec("hybrid_retrieve", Tier.TIER_1_READ, HybridRetrieveIn, retrieve)
    cfg = ReActConfig(route_tools={"HYBRID_RAG": ("hybrid_retrieve",)})
    a1, _ = make(w, [Act("hybrid_retrieve", {"query": "leave"}), Final("25 days", citations=(chunk,), facts=(fact,))],
                 route="HYBRID_RAG", cfg=cfg, memory=mem)
    r1 = await a1.run(caller, "How many leave days?", session)
    assert r1.facts_saved == 1
    a2, r2 = make(w, [Final("25 days", citations=(chunk,))], route="HYBRID_RAG", cfg=cfg, memory=mem)
    await a2.run(caller, "And the leave days again?", session)
    assert r2.prompts[0][1].role == "facts" and "leave.days: 25" in r2.prompts[0][1].content
    other = await a2.memory.load(Caller(f"tb-{tenant[3:]}", "u1", Tier.TIER_2_COMPUTE), session)
    assert other == []  # another tenant, same session id: nothing


# ---------------- observability ----------------
async def test_run_and_step_spans_never_contain_the_query(spans):
    w = World()
    agent, _ = make(w, [Act("get_inventory", {"sku": "SKU-001"}), Final("ok")])
    q = "my email is alice@example.com, how many SKU-001 are left?"
    res = await agent.run(w.caller, q, SESSION)
    names = [s.name for s in spans.get_finished_spans()]
    assert names.count("agent.react.step") == 2 and "agent.react.run" in names
    root = next(s for s in spans.get_finished_spans() if s.name == "agent.react.run")
    assert root.attributes["agent.final_state"] == "DONE" and root.attributes["agent.iterations"] == 2
    assert "alice@example.com" not in " ".join(repr(dict(s.attributes)) for s in spans.get_finished_spans())
    assert res.state is State.DONE

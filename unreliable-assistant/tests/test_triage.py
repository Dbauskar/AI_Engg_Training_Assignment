import numpy as np
import pytest

from agents.triage import TriageAgent
from agents.triage_examples import ROUTE_OF, TRAIN
from evals.triage_set import DEV, TEST
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder
from tests.test_rag_pipeline import ROOT, needs_models


@pytest.fixture(scope="session")
def triage():
    if not (ROOT / DEFAULT_EMBED_DIR / "onnx/model.onnx").exists():
        pytest.skip("models missing")
    return TriageAgent(OnnxEmbedder(ROOT / DEFAULT_EMBED_DIR, threads=2))


def norm(s):
    return " ".join(s.lower().split())


def test_train_dev_and_test_sets_do_not_overlap():
    train = {norm(q) for qs in TRAIN.values() for q in qs}
    for name, data in (("DEV", DEV), ("TEST", TEST)):
        assert train.isdisjoint({norm(q) for q, _ in data}), f"{name} leaks into TRAIN"
    assert {norm(q) for q, _ in DEV}.isdisjoint({norm(q) for q, _ in TEST})


@needs_models
@pytest.mark.parametrize("q", ["Hello", "hi there, good afternoon", "Thanks a lot", "Bye"])
def test_greetings_are_direct_answers_by_rule(triage, q):
    r = triage.classify(q)
    assert (r.route, r.intent, r.source, r.confidence) == ("DIRECT_ANSWER", "chitchat", "rule", 1.0)


@needs_models
def test_transaction_requests_route_with_risk_tier_3(triage):
    r = triage.classify("Transfer 1250.50 USD from ACC-12345678 to ACC-87654321")
    assert (r.route, r.intent, r.risk_tier, r.source) == ("TRANSACTION_EXEC", "transaction", 3, "rule")


@needs_models
def test_policy_question_about_wires_is_not_mistaken_for_a_transaction(triage):
    r = triage.classify("Who has to approve a 300,000 USD wire transfer?")
    assert r.route == "HYBRID_RAG" and r.risk_tier == 1  # a question about policy, not a request to act


@needs_models
def test_out_of_scope_is_rejected_and_in_scope_is_not(triage):
    assert triage.classify("What is the capital of France?").route == "REJECT_OOS"
    r = triage.classify("How many vacation days do I get each year?")
    assert r.route == "HYBRID_RAG" and r.domain == "hr"


@needs_models
def test_complexity_heuristic(triage):
    assert triage.classify("What is the minimum password length?").complexity == "simple"
    assert triage.classify("Compare the refund policy and the warranty policy for batteries").complexity == "complex"
    assert triage.classify("What is the refund window? And what is the warranty length?").complexity == "complex"


@needs_models
def test_classification_is_deterministic(triage):
    a = triage.classify("How often are user access rights reviewed?")
    b = triage.classify("How often are user access rights reviewed?")
    assert (a.route, a.domain, a.confidence) == (b.route, b.domain, b.confidence)


@needs_models
def test_kpi04_routing_accuracy_and_kpi06_false_refusals_on_held_out_set(triage):
    ok = wrong = in_scope = false_refusals = 0
    errors = []
    for q, label in TEST:
        r = triage.classify(q)
        good = r.route == ROUTE_OF[label] and (ROUTE_OF[label] != "HYBRID_RAG" or r.domain == label)
        ok += good
        if not good:
            errors.append((q, label, r.route, r.domain))
        if ROUTE_OF[label] != "REJECT_OOS":
            in_scope += 1
            false_refusals += r.route == "REJECT_OOS"
    acc = ok / len(TEST)
    assert acc >= 0.97, f"KPI-04 {acc:.3f} < 0.97; errors: {errors}"
    assert false_refusals / in_scope <= 0.01  # KPI-06


@needs_models
def test_kpi05_latency_p95_within_25ms(triage):
    for q, _ in TEST[:5]:
        triage.classify(q)  # warm-up
    lat = [triage.classify(q).latency_ms for q, _ in TEST for _ in range(2)]
    assert np.percentile(lat, 95) <= 25, f"p95={np.percentile(lat, 95):.1f} ms"


@needs_models
def test_span_has_route_but_never_the_query(triage, spans):
    q = "my email is alice@example.com, how many leave days do I get?"
    triage.classify(q)
    sp = [s for s in spans.get_finished_spans() if s.name == "triage.classify"]
    assert sp and sp[0].attributes["triage.route"] == "HYBRID_RAG"
    assert "alice@example.com" not in repr(dict(sp[0].attributes))


# Regression (found by the end-to-end API test): questions ABOUT transfers/payments are policy questions, never requests to act.
@needs_models
@pytest.mark.parametrize("q", ["Who has to approve a wire transfer above 10,000 USD?", "What is the limit for sending money to a new beneficiary?",
                               "How do I transfer funds between accounts?", "Which approvals are needed to pay a supplier invoice?",
                               "Is a payment above 5,000 USD allowed without manager approval?", "When can I cancel an order?",
                               "Does a refund above 5,000 USD need manager approval?",
                               "My email is <EMAIL>. Who has to approve a wire above 10,000 USD?",
                               "I want to know who must approve a payment above 5,000 USD?"])
def test_questions_about_transactions_are_never_routed_as_transaction_requests(triage, q):
    assert triage.classify(q).route != "TRANSACTION_EXEC"


@needs_models
@pytest.mark.parametrize("q", ["Transfer 1250.50 USD from ACC-12345678 to ACC-87654321", "I want to move money to another account",
                               "Please set up a payment to a supplier for me", "Can you send 75 GBP to ACC-99887766 today",
                               "Can you send 75 GBP to ACC-99887766 today?"])
def test_genuine_requests_still_route_to_transaction_exec(triage, q):
    assert triage.classify(q).route == "TRANSACTION_EXEC"

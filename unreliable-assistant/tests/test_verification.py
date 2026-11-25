import inspect
import re
import uuid
from pathlib import Path

import numpy as np
import pytest

from agents.nli import DEFAULT_NLI_DIR, OnnxNli
from agents.react import Evidence, Final
from agents.subagents import StateStoreAuditSink
from agents.verification import VerificationSubagent, check_claim_guards, split_claims
from evals.nli_fresh import CLAIMS as FRESH
from evals.nli_set import CLAIMS as SET1
from mcp_servers.rag_pipeline import parse_note
from tests.test_rag_pipeline import ROOT
from tests.test_state_store import store  # noqa: F401

WIRE = "wires-0000-da25f91d"
WIRE_TEXT = ("Every wire transfer above 10,000 USD requires approval from two authorised approvers. Transfers above "
             "250,000 USD additionally require sign-off from the compliance officer. The requester can never approve "
             "their own transfer.")
HAS_NLI = (ROOT / DEFAULT_NLI_DIR / "onnx/model.onnx").exists()
needs_nli = pytest.mark.skipif(not HAS_NLI, reason="NLI model missing: uv run python scripts/fetch_models.py")


class OverlapNli:
    """Deterministic stand-in: entailment = share of the claim's words found in the premise."""
    calls: list = []

    def probs(self, premises, hyps):
        self.calls.append((list(premises), list(hyps)))
        w = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
        ent = [len(w(h) & w(p)) / max(1, len(w(h))) for p, h in zip(premises, hyps)]
        return np.array(ent, dtype=np.float32), np.zeros(len(ent), dtype=np.float32)


def ev(**kw):
    return Evidence(chunk_ids={WIRE}, texts={WIRE: WIRE_TEXT}, tenant_id="acme-corp", user_id="u1", **kw)


class Sink:
    def __init__(self):
        self.events = []

    async def record(self, tenant, actor, event_type, payload):
        self.events.append((tenant, actor, event_type, payload))


def verifier(sink=None):
    return VerificationSubagent(OverlapNli(), sink)


# ---------------- deterministic guards ----------------
def test_split_claims_ignores_citation_markers_and_fragments():
    claims = split_claims(f"Wires above 10,000 USD need two approvers [{WIRE}]. See above. The requester cannot self approve.")
    assert claims == ["Wires above 10,000 USD need two approvers .", "The requester cannot self approve."]  # 'See above.' dropped


def test_guards_catch_invented_numbers_and_unsupported_absolutes():
    assert check_claim_guards("Wires above 10,000 USD need two approvers.", WIRE_TEXT) is None
    assert check_claim_guards("Wires above 20,000 USD need two approvers.", WIRE_TEXT) == "number"
    assert check_claim_guards("Wires need three approvers.", WIRE_TEXT) == "number"  # 'three' is a number word
    assert check_claim_guards("Every wire needs the CEO to sign.", WIRE_TEXT) is None  # 'every' IS in the chunk
    assert check_claim_guards("Wires are always approved instantly.", WIRE_TEXT) == "quantifier"
    # the guard only checks that an absolute WORD is present in the evidence, not what it means: that is NLI's job
    assert check_claim_guards("Approval is never required.", WIRE_TEXT) is None


# ---------------- verdicts ----------------
async def test_supported_answer_passes():
    r = await verifier().verify(Final(f"The requester can never approve their own transfer. [{WIRE}]", (WIRE,)), ev(), "HYBRID_RAG")
    assert r.ok


async def test_invented_figure_is_vetoed_with_a_reason_the_generator_can_use():
    r = await verifier().verify(Final(f"Wires above 50,000 USD need two approvers. [{WIRE}]", (WIRE,)), ev(), "HYBRID_RAG")
    assert not r.ok and "number" in r.reason and "50,000" in r.reason


async def test_unsupported_absolute_is_vetoed():
    r = await verifier().verify(Final("Transfers are always approved on the same day.", (WIRE,)), ev(), "HYBRID_RAG")
    assert not r.ok and "quantifier" in r.reason


async def test_one_bad_sentence_vetoes_an_otherwise_good_answer():
    ans = "The requester can never approve their own transfer. Wires above 90,000 USD need a vote of the board."
    r = await verifier().verify(Final(ans, (WIRE,)), ev(), "HYBRID_RAG")
    assert not r.ok


async def test_citation_rules():
    v = verifier()
    assert not (await v.verify(Final("Two approvers are needed."), ev(), "HYBRID_RAG")).ok  # uncited
    assert (await v.verify(Final("I don't know."), ev(abstained=True), "HYBRID_RAG")).ok  # abstention needs no citation
    assert not (await v.verify(Final("x y z", ("ghost-0000-deadbeef",)), ev(), "HYBRID_RAG")).ok  # never retrieved
    assert (await v.verify(Final("Hello there friend."), ev(), "DIRECT_ANSWER")).ok  # grounding only applies to RAG


# ---------------- independence ----------------
def test_verifier_interface_takes_no_conversation_or_reasoning():
    params = list(inspect.signature(VerificationSubagent.verify).parameters)
    assert params == ["self", "final", "evidence", "route"]  # no prompt, history, goal anchor or thoughts


async def test_nli_sees_only_chunk_sentences_and_claims():
    nli = OverlapNli()
    nli.calls = []
    secret = "SECRET-USER-QUERY-9921"
    v = VerificationSubagent(nli)
    await v.verify(Final("Wires above 10,000 USD need two approvers.", (WIRE,)), ev(), "HYBRID_RAG")
    seen = " ".join(" ".join(p + h) for p, h in nli.calls)
    assert secret not in seen and "approvers" in seen and nli.calls


# ---------------- audit trail ----------------
async def test_verdicts_are_audited_as_counts_and_digests_never_text():
    sink = Sink()
    bad = "Wires above 50,000 USD need two approvers. Transfers are always approved on the same day."
    await verifier(sink).verify(Final(bad, (WIRE,)), ev(), "HYBRID_RAG")
    tenant, actor, kind, payload = sink.events[0]
    assert (tenant, actor, kind) == ("acme-corp", "verification", "verify.verdict")
    assert payload["verdict"] == "veto" and payload["unsupported"] == 2 and len(payload["answer_sha256"]) == 64
    assert "50,000" not in str(payload) and "always approved" not in str(payload)


async def test_audit_failure_never_breaks_verification():
    class Broken:
        async def record(self, *a):
            raise RuntimeError("db down")

    r = await verifier(Broken()).verify(Final("The requester can never approve their own transfer.", (WIRE,)), ev(), "HYBRID_RAG")
    assert r.ok


async def test_verdict_lands_in_the_hash_chained_audit_log(store, pg_test_db):  # noqa: F811
    import psycopg

    tenant = f"ta-{uuid.uuid4().hex[:8]}"
    e = Evidence(chunk_ids={WIRE}, texts={WIRE: WIRE_TEXT}, tenant_id=tenant, user_id="u1")
    v = VerificationSubagent(OverlapNli(), StateStoreAuditSink(store))
    await v.verify(Final("Wires above 50,000 USD need two approvers.", (WIRE,)), e, "HYBRID_RAG")
    assert await store.verify_chain(tenant) == (True, None, 1)
    with psycopg.connect(pg_test_db["admin"]) as c:
        row = c.execute("SELECT event_type, actor, payload FROM audit_log WHERE tenant_id=%s", (tenant,)).fetchone()
    assert row[0] == "verify.verdict" and row[1] == "verification" and row[2]["verdict"] == "veto"
    assert "50,000" not in str(row[2])


# ---------------- real NLI model quality (small sets: indicative, not statistical proof) ----------------
def _bodies():
    return {p.stem: re.sub(r"^# .*\n\n", "", parse_note(p.read_text())[1].strip()) for p in (ROOT / "knowledge_vault").glob("*.md")}


@pytest.fixture(scope="session")
def real_nli():
    if not HAS_NLI:
        pytest.skip("NLI model missing")
    return OnnxNli(ROOT / DEFAULT_NLI_DIR)


@needs_nli
@pytest.mark.parametrize("name,data", [("designed-on", SET1), ("fresh", FRESH)])
def test_real_verifier_never_accepts_an_unsupported_claim_and_rarely_rejects_a_supported_one(real_nli, name, data):
    v, bodies = VerificationSubagent(real_nli), _bodies()
    wrongly_accepted, wrongly_rejected, n_sup = [], [], 0
    for note, claim, supported in data:
        ok = v._judge([claim], bodies[note])[0].supported
        n_sup += supported
        if ok and not supported:
            wrongly_accepted.append(claim)
        if not ok and supported:
            wrongly_rejected.append(claim)
    assert wrongly_accepted == [], f"{name}: unsupported claims accepted: {wrongly_accepted}"  # safety-critical direction
    assert len(wrongly_rejected) <= 0.1 * n_sup, f"{name}: supported claims rejected: {wrongly_rejected}"


def test_list_lead_in_is_not_a_claim_but_a_figure_in_it_is():
    from agents.verification import split_claims
    ans = "Key rules in the Password Policy:\n- Passwords must be at least 14 characters long. [passwords-0000-498b6153]"
    assert split_claims(ans) == ["Passwords must be at least 14 characters long."]
    assert split_claims("The refund limit is 50,000 USD:") == ["The refund limit is 50,000 USD:"]


def test_user_figures_may_be_restated_but_new_figures_still_veto():
    from agents.verification import check_claim_guards
    prem = "Data subject deletion requests must be completed within 30 days of a verified request."
    assert check_claim_guards("A request open for 40 days exceeds that window.", prem, "My request has been open 40 days") is None
    assert check_claim_guards("A request open for 40 days exceeds that window.", prem, "") == "number"
    assert check_claim_guards("It is due in March 2027.", prem, "My request has been open 40 days") == "number"  # invented/derived figure


def test_near_verbatim_restatement_passes_but_changed_meaning_does_not():
    from agents.verification import near_verbatim
    src = ["Access rights are reviewed quarterly by the owning manager."]
    assert near_verbatim("Access rights are reviewed quarterly by the owning manager.", src)
    assert not near_verbatim("Access rights are not reviewed quarterly by the owning manager.", src)  # negation flips meaning

"""Verification subagent: an INDEPENDENT check that a drafted answer is supported by the chunks it cites.

Independence: it receives only (draft answer, retrieved chunk texts, route). It never sees the user's conversation,
the goal anchor, the generator's reasoning or its prompt, and it uses a different model (an NLI cross-encoder), so
a persuasive but wrong draft cannot talk it into approving.

Layered, because NLI alone proved unreliable (it accepted "carry over without limit" from "up to 5 days"):
  1. every cited chunk id must have been retrieved this run;
  2. every number in a claim must appear in the cited chunks (no invented figures);
  3. absolute words in a claim (never, always, every, all, unlimited, ...) must appear in the cited chunks;
  4. NLI: some sentence (or 2-sentence window) of a cited chunk must entail the claim (prob >= threshold).
Only verdict counts and a digest go to the audit trail, never answer text (CLAUDE.md rule 4).
"""
from __future__ import annotations

import asyncio
import hashlib
import difflib
import re
from dataclasses import dataclass
from typing import Any, Protocol

from opentelemetry import trace

from agents.react import Evidence, Final, VerifyResult
from mcp_servers.rag_pipeline import bm25_tokens

_tracer = trace.get_tracer("unreliable_assistant.verification")

ENTAILMENT_THRESHOLD = 0.5
TOP_SENTENCES = 3  # sentences of the cited text actually sent to the NLI model, per claim
CITATION = re.compile(r"\[?\b[a-z0-9]{2,16}-\d{4}-[0-9a-f]{8}\b\]?")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
ABSOLUTE = re.compile(r"\b(never|always|every|all|any|unlimited|without limit|no limit|at any time)\b", re.I)
_WORD_NUM = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


_META = re.compile(r"^(?:the\s+)?(?:passages?|documents?|policy|policies|sources?|context)\s+(?:do(?:es)?\s+not|don't|doesn't|never)\s+"
                   r"(?:state|say|mention|address|name|identify|specify|give|cover|describe|include|contain|provide)\b", re.I)


_ATTRIB = re.compile(r"^(?:(?:the\s+)?(?:passages?|documents?|policy|policies|[\w ]{0,30}?policy)\s+(?:also\s+)?(?:states?|says?|specif(?:y|ies)|requires?\s+that|notes?)\s+that\s+|"
                     r"(?:they|it)\s+(?:also\s+)?(?:states?|says?|specif(?:y|ies)|notes?)\s+(?:that\s+)?|"
                     r"(?:additionally|also|furthermore|moreover),?\s+)", re.I)


def _strip_attribution(claim: str) -> str:
    """'The policy states that X' asserts X; the attribution wrapper itself is not a claim about the world."""
    m = _ATTRIB.match(claim)
    rest = claim[m.end():].strip() if m else claim
    return (rest[:1].upper() + rest[1:]) if rest and rest != claim else claim


def split_claims(answer: str) -> list[str]:
    text = CITATION.sub("", answer)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = [p.strip(" -*•\t") for p in parts if len(re.findall(r"\w+", p)) >= 3]
    # "The passages do not state X" is a statement about the documents, not a fact about the world; it can only under-claim
    out = [c for c in out if not _META.match(c)]
    # a list lead-in ("Key rules in the Password Policy:") asserts nothing; a digit means it might carry a figure, so keep those
    return [_strip_attribution(c) for c in out if not (c.endswith(":") and not NUMBER.search(c))]


def _numbers(text: str) -> set[str]:
    nums = {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}
    nums |= {_WORD_NUM[w] for w in re.findall(r"[a-z]+", text.lower()) if w in _WORD_NUM}
    return nums


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _candidates(claim: str, premise: str, k: int = TOP_SENTENCES) -> list[str]:
    """Most claim-relevant sentences / 2-sentence windows of the premise (cheap lexical prefilter)."""
    sents = _sentences(premise)
    cands = sents + [" ".join(sents[i:i + 2]) for i in range(len(sents) - 1)]
    ct = set(bm25_tokens(claim))
    scored = sorted(cands, key=lambda s: -len(ct & set(bm25_tokens(s))) / (1 + 0.02 * len(s.split())))
    return scored[:k]


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    supported: bool
    reason: str  # ok | number | quantifier | nli


class NliLike(Protocol):
    def probs(self, premises: list[str], hypotheses: list[str]) -> Any: ...


_NEGATION = re.compile(r"\b(?:not|no|never|cannot|can't|isn't|aren't|doesn't|don't|without|neither|nor)\b", re.I)
NEAR_VERBATIM = 0.85


def near_verbatim(claim: str, candidates: list[str]) -> bool:
    """A claim that restates a source sentence almost word for word (>= 85% sequence match, same negation) is supported by
    construction; the small NLI model is unreliable on such trivial paraphrases. Numbers and quantifiers are checked separately."""
    cw = re.findall(r"\w+", claim.lower())
    if len(cw) < 4:
        return False
    for cand in candidates:
        if bool(_NEGATION.search(claim)) != bool(_NEGATION.search(cand)):
            continue
        ratio = difflib.SequenceMatcher(None, cw, re.findall(r"\w+", cand.lower()), autojunk=False).ratio()
        if ratio >= NEAR_VERBATIM:
            return True
    return False


def check_claim_guards(claim: str, premise: str, question: str = "") -> str | None:
    """Deterministic checks. Returns a failure code, or None when the claim passes them.
    Figures the user typed in their own question may be restated; any other number must occur in the cited text."""
    missing = _numbers(CITATION.sub("", claim)) - _numbers(premise) - _numbers(question)
    if missing:
        return "number"
    pl = premise.lower()
    for m in ABSOLUTE.finditer(claim):
        if not re.search(rf"\b{re.escape(m.group(1).lower())}\b", pl):
            return "quantifier"
    return None


class VerificationSubagent:
    def __init__(self, nli: NliLike, audit_sink: Any | None = None, pools: Any | None = None,
                 threshold: float = ENTAILMENT_THRESHOLD, llm_judge: Any | None = None) -> None:
        self.nli, self.sink, self.pools, self.threshold = nli, audit_sink, pools, threshold
        self.llm_judge = llm_judge  # optional second stage for claims only the NLI model rejected (agents/llm_verifier.py)

    def _judge(self, claims: list[str], premise: str, question: str = "") -> list[ClaimVerdict]:  # blocking: run off the event loop
        verdicts: list[ClaimVerdict] = []
        pairs: list[tuple[int, str]] = []  # (claim index, candidate premise)
        for i, claim in enumerate(claims):
            code = check_claim_guards(claim, premise, question)
            if code:
                verdicts.append(ClaimVerdict(claim, False, code))
                continue
            if near_verbatim(claim, _sentences(premise)):
                verdicts.append(ClaimVerdict(claim, True, "ok"))
                continue
            verdicts.append(ClaimVerdict(claim, True, "pending"))
            pairs += [(i, c) for c in _candidates(claim, premise)]
        if pairs:
            ent, _ = self.nli.probs([c for _, c in pairs], [claims[i] for i, _ in pairs])
            best: dict[int, float] = {}
            for (i, _), e in zip(pairs, ent):
                best[i] = max(best.get(i, 0.0), float(e))
            for i, v in enumerate(verdicts):
                if v.reason == "pending":
                    ok = best.get(i, 0.0) >= self.threshold
                    verdicts[i] = ClaimVerdict(v.claim, ok, "ok" if ok else "nli")
        return verdicts

    async def _escalate(self, verdicts: list[ClaimVerdict], premise: str, question: str) -> list[ClaimVerdict]:
        """Ask the LLM judge about claims the NLI model rejected. Number and quantifier guard failures are never escalated,
        and a judge error leaves the claim vetoed (fail closed)."""
        j = self.llm_judge
        if j is None:
            return verdicts
        todo = [i for i, v in enumerate(verdicts) if not v.supported and v.reason == "nli"][: j.max_claims]
        if not todo:
            return verdicts
        results = await asyncio.gather(*[j.supported(verdicts[i].claim, premise, question) for i in todo], return_exceptions=True)
        out = list(verdicts)
        for i, ok in zip(todo, results):
            if ok is True:  # an exception object (judge failure) is never truthy-approved: fail closed
                out[i] = ClaimVerdict(out[i].claim, True, "llm")
        return out

    async def verify(self, final: Final, evidence: Evidence, route: str) -> VerifyResult:
        if route != "HYBRID_RAG":
            return VerifyResult(True)
        with _tracer.start_as_current_span("verification.verify") as span:
            res, verdicts = await self._verify(final, evidence)
            span.set_attribute("verify.ok", res.ok)
            span.set_attribute("verify.claims", len(verdicts))
            span.set_attribute("verify.unsupported", sum(not v.supported for v in verdicts))
            if self.sink is not None and evidence.tenant_id:
                try:
                    await self.sink.record(evidence.tenant_id, "verification", "verify.verdict", {
                        "verdict": "pass" if res.ok else "veto", "claims": len(verdicts),
                        "unsupported": sum(not v.supported for v in verdicts),
                        "reasons": sorted({v.reason for v in verdicts if not v.supported}) or ([res.reason[:40]] if not res.ok else []),
                        "answer_sha256": hashlib.sha256(final.answer.encode()).hexdigest(),
                        "cited": list(final.citations)})
                except Exception:
                    span.set_attribute("verify.audit_failed", True)  # auditing must not break answering
            return res

    async def _verify(self, final: Final, evidence: Evidence) -> tuple[VerifyResult, list[ClaimVerdict]]:
        if not final.citations:
            return VerifyResult(evidence.abstained, "" if evidence.abstained else "answer cites no retrieved chunk"), []
        unknown = [c for c in final.citations if c not in evidence.chunk_ids]
        if unknown:
            return VerifyResult(False, "answer cites a chunk that was never retrieved"), []
        premise = "\n".join(evidence.texts.get(c, "") for c in final.citations)
        claims = split_claims(final.answer)
        if not claims:
            return VerifyResult(True), []
        runner = self.pools.llm.run_sync if self.pools else asyncio.to_thread
        verdicts: list[ClaimVerdict] = await runner(self._judge, claims, premise, evidence.query)
        verdicts = await self._escalate(verdicts, premise, evidence.query)
        bad = [v for v in verdicts if not v.supported]
        if bad:
            first = bad[0]
            return VerifyResult(False, f'claim not supported by the cited chunks ({first.reason}): "{first.claim[:120]}"'), verdicts
        return VerifyResult(True), verdicts

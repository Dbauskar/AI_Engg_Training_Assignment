"""Triage agent: classify a query into (intent, domain, risk_tier, complexity) and pick a route.

Routes: DIRECT_ANSWER | HYBRID_RAG | TRANSACTION_EXEC | REJECT_OOS   (docs/design.md section 3.2)

Deterministic by construction ("temperature 0"): regex rules decide the clear cases in microseconds; everything
else is a nearest-neighbour lookup over embeddings of labelled example utterances (agents/triage_examples.py).
The embedder is injected (anything with `encode(list[str], is_query=bool) -> ndarray`), so a distilled model
can replace this later without touching callers. The query text is never put in spans (CLAUDE.md rule 4).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from opentelemetry import trace

from agents.triage_examples import ROUTE_OF, TRAIN

_tracer = trace.get_tracer("unreliable_assistant.triage")

Route = Literal["DIRECT_ANSWER", "HYBRID_RAG", "TRANSACTION_EXEC", "REJECT_OOS"]
DEFAULT_MIN_SIM = 0.65  # chosen on the DEV set (scripts/eval_triage.py); accuracy drops sharply at 0.70


class Embedder(Protocol):
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


@dataclass(frozen=True)
class TriageResult:
    intent: str          # question | transaction | chitchat | out_of_scope
    domain: str          # compliance | security | support | finance | engineering | hr | general
    risk_tier: int       # 1 read, 3 mutate (state-changing request)
    complexity: str      # simple | complex
    route: Route
    confidence: float    # 1.0 for rule hits, else top similarity
    source: str          # rule | embedding
    latency_ms: float


# ---- rules ----------------------------------------------------------------------------------------------------
_GREETING = re.compile(
    r"^\s*(hi|hello|hey|hiya|good\s+(morning|afternoon|evening)|thanks|thank\s+you|thx|cheers|bye|goodbye)\b"
    r"[\s\w!.,']{0,30}$", re.I)
_QUESTION_START = re.compile(r"^\s*(who|what|when|where|which|why|how|is|are|does|do|did|should|must|may|will)\b", re.I)
_REQUEST_START = re.compile(
    r"^\s*(please\s+)?((can|could|would)\s+you\s+(please\s+)?|i\s+(want|need|would\s+like|'d\s+like|wish)\s+to\s+)?"
    r"(transfer|wire|send|pay|remit|order|purchase|cancel|close|reverse|refund|change|update|set|place|submit|move)\b", re.I)
_MONEY_OR_ACCOUNT = re.compile(r"\bACC-\d{8}\b|\b\d[\d,]*(\.\d{1,2})?\s*(usd|eur|gbp|dollars?|euros?|pounds?)\b|\border\s+\d+", re.I)
_COMPLEX = re.compile(r"\b(compare|versus|vs\.?|difference between|both|as well as|step[- ]by[- ]step|and also)\b", re.I)


def _is_question(q: str) -> bool:
    """Any sentence opening with an interrogative ('My email is X. Who approves wires?'), or a closing '?' on something that
    is not phrased as a request ('Can you send ...' is a request even with a question mark)."""
    if any(_QUESTION_START.match(sent) for sent in re.split(r"(?<=[.!?])\s+", q.strip())):
        return True
    return q.rstrip().endswith("?") and not _REQUEST_START.match(q)


def _complexity(q: str) -> str:
    return "complex" if (len(q.split()) > 25 or q.count("?") >= 2 or _COMPLEX.search(q)) else "simple"


def _result(label: str, complexity: str, conf: float, source: str, t0: float) -> TriageResult:
    route = ROUTE_OF[label]
    intent = {"HYBRID_RAG": "question", "TRANSACTION_EXEC": "transaction",
              "DIRECT_ANSWER": "chitchat", "REJECT_OOS": "out_of_scope"}[route]
    domain = label if route == "HYBRID_RAG" else ("finance" if route == "TRANSACTION_EXEC" else "general")
    return TriageResult(intent, domain, 3 if route == "TRANSACTION_EXEC" else 1, complexity,
                        route, conf, source, (time.perf_counter() - t0) * 1000)  # type: ignore[arg-type]


class TriageAgent:
    def __init__(self, embedder: Embedder, min_sim: float = DEFAULT_MIN_SIM,
                 examples: dict[str, list[str]] | None = None) -> None:
        self.min_sim = min_sim
        ex = examples or TRAIN
        self._labels: list[str] = []
        texts: list[str] = []
        for label, items in ex.items():
            self._labels += [label] * len(items)
            texts += items
        self._emb = embedder
        self._vecs = self._norm(np.asarray(embedder.encode(texts, False), dtype=np.float32))
        self._label_names = sorted(set(self._labels))
        self._label_idx = np.array([self._label_names.index(label) for label in self._labels])

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)

    def classify(self, query: str) -> TriageResult:
        with _tracer.start_as_current_span("triage.classify") as span:
            res = self._classify(query)
            span.set_attribute("triage.route", res.route)
            span.set_attribute("triage.source", res.source)
            span.set_attribute("triage.domain", res.domain)
            span.set_attribute("triage.latency_ms", res.latency_ms)
            return res

    def _classify(self, query: str) -> TriageResult:
        t0 = time.perf_counter()
        q = query.strip()
        cx = _complexity(q)
        if len(q.split()) <= 8 and _GREETING.match(q):
            return _result("chitchat", cx, 1.0, "rule", t0)
        if _REQUEST_START.match(q) and not _QUESTION_START.match(q) and _MONEY_OR_ACCOUNT.search(q):
            return _result("transaction", cx, 1.0, "rule", t0)

        v = self._norm(np.asarray(self._emb.encode([q], False), dtype=np.float32))[0]
        sims = self._vecs @ v
        best_per_label = np.full(len(self._label_names), -1.0, dtype=np.float32)
        np.maximum.at(best_per_label, self._label_idx, sims)  # nearest neighbour within each label
        if _is_question(q):  # "Who has to approve a wire...?" asks ABOUT a transfer; it is not a request to make one
            best_per_label[self._label_names.index("transaction")] = -1.0
        j = int(best_per_label.argmax())
        label, sim = self._label_names[j], float(best_per_label[j])
        if sim < self.min_sim:
            label = "oos"  # not close enough to anything we serve
        return _result(label, cx, sim, "embedding", t0)

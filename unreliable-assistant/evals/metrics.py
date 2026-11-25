"""Evaluation metrics as small pure functions (unit-tested against hand-computed values in tests/test_metrics.py).

Retrieval metrics work on ranked DOCUMENT ids and a set of gold document ids (binary relevance).
Definitions follow the usual IR conventions; context_precision follows RAGAS (average precision over ranks).
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

_MONTHS = ("january february march april may june july august september october november december").split()


def mrr_at_k(ranked: Sequence[str], gold: set[str], k: int = 4) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int = 4) -> float:
    """Binary-gain nDCG: DCG = sum rel_i / log2(i+1); ideal puts min(k, |gold|) relevant documents first."""
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 1.0


def context_precision(ranked: Sequence[str], gold: set[str], k: int = 4) -> float:
    """RAGAS context precision: mean of precision@i taken at every rank i that holds a relevant item. 0 if none is relevant."""
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(",", "")).strip()


def context_recall(context: str, gold_facts: Iterable[str]) -> float:
    """Fraction of gold evidence sentences that appear in the retrieved context (the reference evidence is the vault text itself)."""
    facts = [f for f in gold_facts if f.strip()]
    if not facts:
        return 1.0
    ctx = _norm(context)
    return sum(_norm(f.rstrip(".")) in ctx for f in facts) / len(facts)


def extract_entities(text: str) -> set[str]:
    """Deterministic entity set: numbers, ids (SKU-205, ACC-...), acronyms, month names, and Capitalised phrases that do not start a sentence."""
    t = text.replace(",", "")
    ents = {n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", t)}
    ents |= {m.lower() for m in re.findall(r"\b[A-Z]{2,}-\d+\b", t)}
    ents |= {m.lower() for m in re.findall(r"\b[A-Z]{2,}\b", t)}
    ents |= {m for m in re.findall(r"[a-z]+", text.lower()) if m in _MONTHS}
    for sent in re.split(r"(?<=[.!?])\s+", text):
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", sent)
        phrase: list[str] = []
        for w in words[1:] + [""]:  # the first word of a sentence is capitalised by convention, so skip it
            if w[:1].isupper() and not w.isupper():
                phrase.append(w)
            elif phrase:
                ents.add(" ".join(phrase).lower())
                phrase = []
    return ents


def context_entity_recall(reference: str, context: str) -> float:
    """Fraction of the reference evidence's entities that the retrieved context contains."""
    ents = extract_entities(reference)
    if not ents:
        return 1.0
    ctx = _norm(context)
    return sum(e in ctx for e in ents) / len(ents)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

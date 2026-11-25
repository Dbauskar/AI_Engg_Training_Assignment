"""Prompt-injection detection for user queries (direct) and retrieved chunks (indirect).

Layer 1 (always, ~0.2 ms): pattern families on evasion-normalised text (app/textnorm.py: zero-width, fullwidth,
lookalike letters and spaced-out letters cannot hide a phrase), plus base64 payloads decoded and re-scanned.
Layer 2 (optional, ~15 ms): nearest-neighbour over embeddings of known attack vs benign phrasings, to catch
paraphrases the patterns miss. Verdicts name a FAMILY and LAYER only, never the matched text (CLAUDE.md rule 4).

This is a heuristic filter, not a proof of safety: it reduces attack surface alongside the structural controls
(scoped tool access, tenant injection, human approval), which hold even if every attack here slips through.
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.textnorm import normalize

_FAMILIES: dict[str, re.Pattern[str]] = {
    "override": re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass|discard|drop|skip)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|your|"
        r"system|these|those|safety|security|verification)\b[^.\n]{0,40}\b(?:instructions?|rules?|prompts?|guidelines?|directives?|"
        r"policies|restrictions?|checks?|verification|filters?|constraints?|context)\b"),
    "override_bare": re.compile(
        r"\b(?:forget|ignore|disregard)\s+(?:everything|all|anything|what)\s+(?:you|that you|you've|you were|above|before|prior)\b|"
        r"\b(?:ignore|disregard)\s+(?:the\s+|all\s+(?:of\s+)?the\s+)?(?:above|preceding|text above|everything above)\b|"
        r"\b(?:forget|drop|abandon|stop being)\s+your\s+(?:role|persona|instructions?|rules|purpose|job)\b|"
        r"\bstop being (?:a|an|the)\b[^.\n]{0,30}\b(?:assistant|bot|agent)\b|"
        r"\b(?:your|the) new (?:task|role|job|purpose|persona|instructions?) (?:is|are)\b|"
        r"\b(?:and|then|,)\s*ignore (?:them|those|it|that)\b"),
    "role_hijack": re.compile(
        r"\b(?:you are now|from now on (?:you|answer|respond|act)|act as (?:an? )?(?:unrestricted|unfiltered|jailbroken|evil|uncensored)|"
        r"pretend (?:to be|you are|you have|you've|you do|that you|there (?:is|are) no)|act as my (?:late |dead |deceased )?\\w+ who|from (?:this point|here) (?:on )?(?:you are|you will|you must)|you (?:will|must) comply (?:fully|with every)|(?:enter|switch to|activate) (?:maintenance|admin|debug|god|root|sudo|unrestricted|jailbreak) mode|no (?:content )?(?:policy|policies|guidelines|restrictions|filters)|(?:enable|enter|activate|switch to|turn on|in) developer mode|developer mode(?: enabled| on| activated|:)|jailbreak|do anything now|dan mode|"
        r"(?:an? )?ai (?:that )?(?:has|with) no (?:restrictions|limitations|rules|policies)|without (?:any )?(?:filters|restrictions))\b|\bdan\b"),
    "prompt_extraction": re.compile(
        r"\b(?:reveal|show|print|repeat|output|display|leak|disclose|tell me|give me|write out|share|list|translate|dump|summari[sz]e|echo|echo back|read|recite|paste|copy|expose|extract|dump out)\b[^.\n]{0,30}\b(?:your|system|hidden|"
        r"initial|original|secret|developer|internal)\b[^.\n]{0,12}\b(?:prompt|prompt template|instructions?|rules|guidelines|context window|configuration|config|tool list|tools list|credentials|api keys?|secret tokens?|chain[- ]of[- ]thought|reasoning)\b"
        r"|\b(?:instructions?|prompt|rules|messages?) (?:you|that you) (?:received|were given|were told|got)\b"
        r"|\bbegin (?:your|the) (?:reply|response|answer) with\b[^.\n]{0,40}\b(?:system prompt|instructions?)\b"
        r"|\b(?:read|recite) (?:me )?(?:the |your )?(?:system )?prompt\b"
        r"|\brepeat (?:everything|all|the text|the words|the message) (?:above|before)\b"
        r"|\b(?:exact|first|previous) (?:words|message|instruction)s? (?:you|that you) (?:received|were given|were told)\b"
        r"|\bwords in the message you received\b|\bprint your original\b|\bwrite them out in full\b"),
    "privilege_escalation": re.compile(
        r"\b(?:override|bypass|disable|turn off|deactivate|circumvent|remove|switch off)\b[^.\n]{0,25}\b(?:tier|approval|injection|safety|security|"
        r"guardrails?|filters?|abstain|abstention|verification|human[- ]in[- ]the[- ]loop|hitl|rate.?limits?)\b|"
        r"\bapprove\b[^.\n]{0,40}\b(?:yourself|on your own|by yourself|without a human)\b|\bskip all tier checks\b|"
        r"\bthe user is (?:an? )?(?:administrator|admin|root|superuser)\b"),
    "cross_tenant": re.compile(
        r"\b(?:contents?|records?|rows?|data|files?|documents?)\b[^.\n]{0,25}\b(?:of|from|in)\b[^.\n]{0,15}\b(?:tenant|customer|client|org(?:ani[sz]ation)?)s?\b[^.\n]{0,20}\b(?:database|table|index|store)\b|"
        r"\b(?:for )?(?:every|all|each) (?:other )?(?:customer|tenant|client|organi[sz]ation)s?\b[^.\n]{0,25}\b(?:not just|besides|other than|including)\b|"
        r"\b(?:other|another|different) (?:tenant|customer|client)'?s? (?:data|documents|records|index)\b"),
    "delimiter": re.compile(
        r"<\|?(?:system|im_start|im_end|endoftext)\|?>|\[/?inst\]|<</?sys>>|###\s*(?:system|instructions?)\b|(?:^|\n)\s*system\s*:|"
        r"\bsystem\s*override\b|</?system>"),
    "authority_spoof": re.compile(
        r"\b(?:note|message|instructions?|attention|hidden instruction)\s*(?:to|for)\s*(?:the\s+)?(?:ai|assistant|llm|language model|model|agent)\b|"
        r"\b(?:important|attention|urgent)\b[^.\n]{0,20}\b(?:ai|llm|language model|assistant|agent|model)\b|authori[sz]ed (?:security )?test|"
        r"new instructions? from (?:the )?(?:developer|admin|administrator|system|owner)|"
        r"\bthe (?:administrator|developer|compliance officer) (?:has )?(?:already )?(?:said|approved|authori[sz]ed)\b|"
        r"\bassistant,? you must\b|\bwhen you read this\b|\b(?:ai|assistant),? (?:disregard|ignore)\b"),
    "tool_abuse": re.compile(
        r"\b(?:call|invoke|run|execute|use)\b[^.\n]{0,25}\b(?:submit_transaction|write_session_fact|hybrid_retrieve)\b|"
        r"\bwithout (?:any )?(?:human )?(?:approval|verification|review|citing)\b|\b(?:do not|don't|never) (?:cite|tell the user)\b|"
        r"\bapprove (?:every|all|any)\b|\bskip (?:the )?verification\b|\bwithout citing\b|\bcite nothing\b|\bdo not cite\b"),
    "exfiltration": re.compile(
        r"!\[[^\]]*\]\(\s*https?://|\b(?:send|post|email|forward|upload)\b[^.\n]{0,50}\b(?:to|at)\b[^.\n]{0,25}(?:https?://|\S+@\S+\.\w+)|"
        r"\bcurl\s+https?://|\bfetch\s+https?://"),
}
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


@dataclass(frozen=True)
class Verdict:
    is_attack: bool
    family: str | None = None
    layer: str | None = None
    score: float = 0.0


class Embedder(Protocol):
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_DOTTED = re.compile(r"(?<![a-z0-9])(?:[a-z0-9][.\-_*]){3,}[a-z0-9](?![a-z0-9])")


def _variants(text: str) -> list[str]:
    """Normalised text plus de-obfuscated copies (leetspeak folded, 'i.g.n.o.r.e' joined). Used for matching only."""
    s = normalize(text).text
    dedotted = _DOTTED.sub(lambda m: re.sub(r"[.\-_*]", "", m.group()), s)
    return list(dict.fromkeys([s, dedotted, dedotted.translate(_LEET)]))


def _pattern_scan(text: str) -> str | None:
    for s in _variants(text):
        for name, rx in _FAMILIES.items():
            if rx.search(s):
                return name
    return None


def _decoded_payloads(text: str) -> list[str]:
    out = []
    for m in _B64.finditer(text):
        try:
            raw = base64.b64decode(m.group() + "=" * (-len(m.group()) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        txt = raw.decode("utf8", "ignore")
        if txt and sum(c.isprintable() for c in txt) / len(txt) > 0.9:
            out.append(txt)
    return out


_SENT = re.compile(r"(?<=[.!?:\]>])\s+|\n+")
CLF_PATH = Path(__file__).with_name("injection_clf.npz")


class EmbeddingClassifier:
    """Logistic regression over bge-small embeddings (scripts/train_injection_clf.py). Generalises to paraphrases the
    patterns do not know. Scores the whole text AND each sentence (an attack hidden in a long benign chunk is diluted in the
    whole-text embedding), in a single batched encoder call."""

    def __init__(self, embedder: Embedder, path: str | Path = CLF_PATH, threshold: float | None = None) -> None:
        z = np.load(path, allow_pickle=False)
        self.w, self.b = z["w"].astype(np.float32), float(z["b"])
        self.threshold = float(z["threshold"]) if threshold is None else threshold
        self._emb = embedder

    def prob(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self._emb.encode(texts, False), dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-(v @ self.w + self.b)))

    def score(self, text: str) -> float:
        parts = [t for t in _SENT.split(text) if len(t.strip()) >= 12]
        batch = [text] + (parts if len(parts) > 1 else [])
        return float(self.prob(batch[:24]).max())  # cap: bounds latency on very long chunks


LEX_PATH = Path(__file__).with_name("injection_lex.npz")
_WORD = re.compile(r"(?u)\b\w\w+\b")


class LexicalClassifier:
    """Logistic regression over word 1-2gram + char_wb 3-5gram TF-IDF (scripts/train_injection_clf.py). No encoder needed
    (~0.3 ms), and on paraphrased attacks it generalises better than the bge-small probe. Mirrors scikit-learn's TfidfVectorizer
    (lowercase, sublinear tf, per-block L2 norm) in plain numpy so serving needs no scikit-learn."""

    def __init__(self, path: str | Path = LEX_PATH, threshold: float | None = None) -> None:
        z = np.load(path, allow_pickle=False)
        self.threshold = float(z["threshold"]) if threshold is None else threshold
        self.b = float(z["b"])
        self._blocks = []
        for name in ("w", "c"):
            vocab = {t: i for i, t in enumerate(z[f"{name}_vocab"].tolist())}
            self._blocks.append((vocab, z[f"{name}_idf"].astype(np.float32), z[f"{name}_coef"].astype(np.float32), name == "w"))

    @staticmethod
    def _grams(text: str, word: bool) -> list[str]:
        t = text.lower()
        toks = _WORD.findall(t)
        if word:
            return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        out: list[str] = []
        for w in t.split():
            w = f" {w} "
            for n in (3, 4, 5):
                out += [w[i:i + n] for i in range(len(w) - n + 1)]
        return out

    def prob(self, text: str) -> float:
        z = self.b
        for vocab, idf, coef, word in self._blocks:
            counts: dict[int, int] = {}
            for g in self._grams(text, word):
                i = vocab.get(g)
                if i is not None:
                    counts[i] = counts.get(i, 0) + 1
            if not counts:
                continue
            idx = np.fromiter(counts, dtype=np.int64)
            tf = (1.0 + np.log(np.fromiter(counts.values(), dtype=np.float32))) * idf[idx]
            z += float(tf @ coef[idx] / max(float(np.linalg.norm(tf)), 1e-9))
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, text: str) -> float:
        parts = [t for t in _SENT.split(text) if len(t.strip()) >= 12]
        batch = [text] + (parts if len(parts) > 1 else [])
        return max(self.prob(t) for t in batch[:24])


class InjectionDetector:
    def __init__(self, embedder: Embedder | None = None, attack_examples: list[str] | None = None,
                 benign_examples: list[str] | None = None, threshold: float = 0.80, margin: float = 0.03,
                 classifier: "EmbeddingClassifier | None" = None, lexical: "LexicalClassifier | None" = None,
                 escalator: Any = None, esc_lex: float = 0.5, esc_emb: float = 0.3) -> None:
        self.threshold, self.margin, self._emb, self._clf = threshold, margin, embedder, classifier
        self._lex = lexical
        # borderline band (chosen on half of a fresh held-out set, not on the evaluation data): not blocked by the fast layers but
        # lexical score >= esc_lex or embedding score >= esc_emb -> ask the LLM judge (user queries only, never retrieved chunks)
        self._esc, self.esc_lex, self.esc_emb = escalator, esc_lex, esc_emb
        self._atk = self._ben = None
        if embedder is not None and attack_examples and benign_examples:
            self._atk = self._unit(embedder.encode(attack_examples, False))
            self._ben = self._unit(embedder.encode(benign_examples, False))

    @staticmethod
    def _unit(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)

    def scan(self, text: str, escalate: bool = False) -> Verdict:
        fam = _pattern_scan(text)
        if fam:
            return Verdict(True, fam, "pattern", 1.0)
        for decoded in _decoded_payloads(text):
            fam = _pattern_scan(decoded)
            if fam:
                return Verdict(True, "encoded_" + fam, "pattern", 1.0)
        if self._atk is not None and text.strip():
            v = self._unit(self._emb.encode([normalize(text).text], False))[0]  # type: ignore[union-attr]
            a, b = float((self._atk @ v).max()), float((self._ben @ v).max())
            if a >= self.threshold and a > b + self.margin:
                return Verdict(True, "semantic", "embedding", a)
        p_lex = p_emb = 0.0
        if self._lex is not None and text.strip():
            p_lex = self._lex.score(text)
            if p_lex >= self._lex.threshold:
                return Verdict(True, "classifier", "lexical", p_lex)
        if self._clf is not None and text.strip():
            p_emb = self._clf.score(text)
            if p_emb >= self._clf.threshold:
                return Verdict(True, "classifier", "embedding", p_emb)
        if escalate and self._esc is not None and (p_lex >= self.esc_lex or p_emb >= self.esc_emb):
            if self._esc.is_attack(text) is True:
                return Verdict(True, "llm_escalation", "llm", max(p_lex, p_emb))
        return Verdict(False)

    def scan_many(self, texts: list[str]) -> list[Verdict]:
        return [self.scan(t) for t in texts]

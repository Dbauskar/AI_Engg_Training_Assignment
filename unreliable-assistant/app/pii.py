"""PII detection and redaction (CLAUDE.md rule 4; design Gate 13 'zero raw PII persistence').

Two layers, merged:
  1. regex layer on an evasion-normalised copy of the text (app/textnorm.py): emails incl. name[at]domain, (at)/(dot),
     'at ... dot', spaced-out, fullwidth and lookalike-letter forms; phones; SSNs; Luhn-checked cards; IPs;
  2. Microsoft Presidio (spaCy NER on a lean pipeline) for person names, IBANs and a second opinion.
Findings are reported in ORIGINAL-text coordinates, so redaction removes exactly what the user typed.
The module never logs or returns matched text: only types, counts and offsets.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from app.textnorm import normalize

# ---- regex layer -----------------------------------------------------------------------------------------------------
_AT = r"(?:\s*@\s*|\s*[\[\(\{<]\s*at\s*[\]\)\}>]\s*|\s+at\s+)"
_DOT = r"(?:\s*\.\s*|\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*|\s+dot\s+)"
_LOCAL = r"[a-z0-9][a-z0-9._%+\-]{0,62}"
_LABEL = r"[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?"
_TLD = (r"(?:com|org|net|edu|gov|io|co|uk|de|fr|in|us|ca|au|info|biz|me|ai|dev|app|se|nl|es|it|ng|ch|jp|cn|ru|br|za|"
        r"mx|eu|tv|xyz|online|site)")
EMAIL_STD = re.compile(rf"(?<![a-z0-9._%+\-]){_LOCAL}@{_LABEL}(?:\.{_LABEL})*\.[a-z]{{2,}}")
EMAIL_OBF = re.compile(rf"(?<![a-z0-9._%+\-]){_LOCAL}{_AT}{_LABEL}(?:{_DOT}{_LABEL})*{_DOT}{_TLD}(?![a-z0-9])")
PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().\-]{7,22}\d(?![\w])")
SSN = re.compile(r"(?<!\d)\d{3}[\s.\-]\d{2}[\s.\-]\d{4}(?!\d)")
SSN_CTX = re.compile(r"(?:ssn|social security(?: number)?)\D{0,12}(\d{9})(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")

_ID_LIKE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\b[0-9a-f]{32,}\b|\b[a-z0-9]{2,16}-\d{4}-[0-9a-f]{8}\b")
PRIORITY = {"CARD": 6, "SSN": 5, "EMAIL": 4, "IBAN": 4, "PHONE": 3, "IP": 3, "PERSON": 2}
_PRESIDIO_MAP = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "US_SSN": "SSN",
                 "CREDIT_CARD": "CARD", "IBAN_CODE": "IBAN", "IP_ADDRESS": "IP"}


# NER fires on Title Case phrases. spaCy gives every entity the same score, so a threshold cannot separate real names from
# titles like "Wire Transfer Approval Policy"; a PERSON span containing any of these business words is not a name.
NON_NAME_WORDS = frozenset("""policy policies transfer transfers wire wires approval approvals approver request requests
account accounts refund refunds warranty password passwords vendor vendors incident incidents change changes management
retention expense expenses access control remote work paid time off travel data risk hardware software licence license
security compliance freeze release procedure process guide manual standard report review audit plan tier service
support customer customers employee employees leave holiday annual sick""".split())


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


@dataclass(frozen=True)
class Finding:
    type: str
    start: int
    end: int
    source: str  # regex | presidio


@dataclass(frozen=True)
class RedactResult:
    text: str
    counts: dict[str, int]
    findings: tuple[Finding, ...]


class PiiDetected(Exception):
    """Raised by persistence guards. Carries only types and counts, never the matched text."""

    def __init__(self, counts: dict[str, int]) -> None:
        super().__init__("raw PII must not be persisted: " + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())))
        self.counts = counts


def _regex_findings(text: str) -> list[Finding]:
    n = normalize(text)
    s, out = n.text, []

    def add(t: str, m: re.Match, group: int = 0) -> None:
        a, b = m.start(group), m.end(group)
        if a < b:
            oa, ob = n.span_to_original(a, b)
            out.append(Finding(t, oa, ob, "regex"))

    for rx in (EMAIL_STD, EMAIL_OBF):
        for m in rx.finditer(s):
            add("EMAIL", m)
    for m in SSN.finditer(s):
        add("SSN", m)
    for m in SSN_CTX.finditer(s):
        add("SSN", m, 1)
    for m in CARD.finditer(s):
        digits = re.sub(r"\D", "", m.group())
        grouped = re.fullmatch(r"\d{4}([ \-])\d{4}\1\d{4}\1\d{4}", m.group()) is not None
        amex = re.fullmatch(r"\d{4} \d{6} \d{5}", m.group()) is not None
        if 13 <= len(digits) <= 19 and (luhn_ok(digits) or grouped or amex):
            add("CARD", m)
    for m in PHONE.finditer(s):
        raw = m.group()
        digits = re.sub(r"\D", "", raw)
        has_sep = bool(re.search(r"[\s().\-]", raw)) or raw.startswith("+")
        if 9 <= len(digits) <= 15 and (has_sep or len(digits) >= 10) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            add("PHONE", m)
    for m in IPV4.finditer(s):
        add("IP", m)
    return out


# ---- Presidio layer ----------------------------------------------------------------------------------------------------
def _build_presidio() -> Any:
    import spacy
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import SpacyNlpEngine

    class LeanSpacyNlpEngine(SpacyNlpEngine):
        """Same engine, but only tok2vec + NER: ~11 ms instead of ~16 ms per call (we never need parse/lemmas)."""

        def load(self) -> None:
            self.nlp = {"en": spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "tagger", "attribute_ruler"])}

    engine = LeanSpacyNlpEngine(models=[{"lang_code": "en", "model_name": "en_core_web_sm"}])
    return AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])


class PiiDetector:
    def __init__(self, use_ner: bool = True, ner_threshold: float = 0.6) -> None:
        self.use_ner, self.threshold = use_ner, ner_threshold
        self._presidio: Any = None
        self._lock = threading.Lock()  # spaCy/Presidio inference is not documented as thread-safe: serialise NER calls

    def _analyzer(self) -> Any:
        if self._presidio is None:
            self._presidio = _build_presidio()
        return self._presidio

    def find(self, text: str, ner: bool | None = None) -> list[Finding]:
        found = _regex_findings(text)
        if (self.use_ner if ner is None else ner and self.use_ner) and text.strip():
            with self._lock:
                results = self._analyzer().analyze(text=text, language="en", entities=list(_PRESIDIO_MAP),
                                                   score_threshold=self.threshold)
            for r in results:
                kind = _PRESIDIO_MAP[r.entity_type]
                if kind == "PERSON" and NON_NAME_WORDS & set(re.findall(r"[a-z]+", text[r.start:r.end].lower())):
                    continue
                found.append(Finding(kind, r.start, r.end, "presidio"))
        return self._merge(found)

    @staticmethod
    def _merge(found: list[Finding]) -> list[Finding]:
        merged: list[Finding] = []
        for f in sorted(found, key=lambda f: (f.start, -f.end)):
            if merged and f.start <= merged[-1].end:  # overlapping/adjacent: union, keep the more specific type
                p = merged[-1]
                t = p.type if PRIORITY.get(p.type, 0) >= PRIORITY.get(f.type, 0) else f.type
                merged[-1] = Finding(t, p.start, max(p.end, f.end), p.source)
            else:
                merged.append(f)
        return merged

    def redact(self, text: str) -> RedactResult:
        findings = self.find(text)
        out, counts = text, {}
        for f in sorted(findings, key=lambda f: -f.start):
            out = out[:f.start] + f"<{f.type}>" + out[f.end:]
            counts[f.type] = counts.get(f.type, 0) + 1
        return RedactResult(out, counts, tuple(findings))

    def scan(self, obj: Any, ner: bool | None = None, ignore_ids: bool = False) -> dict[str, int]:
        """Count PII findings (by type) in every string inside a nested dict/list. Never returns the text.
        ignore_ids blanks UUIDs / hex digests / chunk ids first (their digit runs look like phone numbers)."""
        counts: dict[str, int] = {}

        def walk(x: Any) -> None:
            if isinstance(x, str):
                for f in self.find(_ID_LIKE.sub(" ", x) if ignore_ids else x, ner):
                    counts[f.type] = counts.get(f.type, 0) + 1
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)

        walk(obj)
        return counts

    def assert_clean(self, obj: Any, ner: bool | None = None, ignore_ids: bool = False) -> None:
        counts = self.scan(obj, ner, ignore_ids)
        if counts:
            raise PiiDetected(counts)


_shared: PiiDetector | None = None


def shared_detector() -> PiiDetector:
    global _shared
    if _shared is None:
        _shared = PiiDetector()
    return _shared

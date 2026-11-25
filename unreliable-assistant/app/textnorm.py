"""Evasion-resistant text normalisation with an index map back to the original string.

Detectors (PII, prompt injection) run on a normalised copy so that zero-width characters, fullwidth forms,
lookalike letters (Cyrillic/Greek) and spaced-out text cannot hide a match. `index_map[i]` is the position in the
ORIGINAL text of normalised character i, so a match can be redacted in the original.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ZERO_WIDTH = set("​‌‍⁠﻿­‎‏‪‫‬‭‮⁦⁧⁨⁩")
# lookalikes -> latin, applied after NFKC and lower-casing
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ѕ": "s", "ј": "j",
    "к": "k", "м": "m", "н": "h", "т": "t", "ԁ": "d", "ɡ": "g", "һ": "h", "ԛ": "q", "ԝ": "w",
    "α": "a", "ο": "o", "ε": "e", "ν": "v", "ρ": "p", "τ": "t", "ι": "i", "κ": "k", "υ": "u", "χ": "x",
    "ı": "i", "ℓ": "l", "ǀ": "l",
})
_SPACED_RUN = re.compile(r"(?<!\S)(?:\S ){4,}\S(?!\S)")  # >=5 single characters separated by single spaces


@dataclass(frozen=True)
class Normalized:
    text: str
    index_map: list[int]

    def span_to_original(self, start: int, end: int) -> tuple[int, int]:
        """[start, end) in normalised coordinates -> [start, end) in the original text."""
        return self.index_map[start], self.index_map[end - 1] + 1


def normalize(text: str, collapse_spaced: bool = True) -> Normalized:
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        for c in unicodedata.normalize("NFKC", ch).lower().translate(_CONFUSABLES):
            out.append(c)
            idx.append(i)
    s = "".join(out)
    if collapse_spaced and _SPACED_RUN.search(s):
        chars, imap, pos = [], [], 0
        for m in _SPACED_RUN.finditer(s):
            chars += list(s[pos:m.start()])
            imap += idx[pos:m.start()]
            for k in range(m.start(), m.end()):
                if s[k] != " ":
                    chars.append(s[k])
                    imap.append(idx[k])
            pos = m.end()
        chars += list(s[pos:])
        imap += idx[pos:]
        s, idx = "".join(chars), imap
    return Normalized(s, idx)

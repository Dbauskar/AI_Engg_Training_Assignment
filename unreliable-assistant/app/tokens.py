"""Model-agnostic token estimate shared by the agent loop and the graph context builder (words + punctuation)."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Deliberately conservative estimate; real tokenisers differ by model, so treat ratios as indicative."""
    return len(_TOKEN.findall(text))


def truncate_tokens(text: str, max_tokens: int, marker: str = "…") -> str:
    """Cut `text` so that count_tokens(result) <= max_tokens, marker included."""
    if count_tokens(text) <= max_tokens:
        return text
    toks = list(_TOKEN.finditer(text))
    cut = toks[max(0, max_tokens - count_tokens(marker)) - 1].end() if max_tokens > 1 else 0
    return text[:cut].rstrip() + marker

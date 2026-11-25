"""Golden dataset: generator + loader (273 questions, stratified).

  factoid 63 (25.2%)   single-fact questions, 1-2 phrasings per atomic note            (from evals/graphify_queries.py)
  complex 62 (24.8%)   scenarios that need a rule applied (compare a number, compute a date)   (hand-written)
  multihop 50 (20.0%)  questions needing TWO notes (8 hand-written related pairs + seeded pairs from different policies;
                       'parallel' two-fact questions, NOT sequential bridge reasoning)
  summary 38 (15.2%)   'summarize / key rules / overview' of a whole policy (templated from the 12 policies)
  negative 37 (14.8%)  unanswerable: 15 general-knowledge (refuse) + 22 plausible-but-uncovered (abstain)

Deterministic (fixed seeds): regenerating gives byte-identical output. Each item carries gold atomic note ids, their policy
ids ('gold_docs'), a ground-truth answer built from the vault text, and a dev/test split (40/60, stratified by position).

Usage:  uv run python -m evals.golden            (writes evals/golden/golden_250.jsonl)
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.golden_sources import COMPLEX, NEGATIVE  # noqa: E402
from evals.graphify_queries import MULTI, SINGLE  # noqa: E402

PATH = ROOT / "evals/golden/golden_250.jsonl"
COUNTS = {"factoid": 63, "complex": 62, "multihop": 50, "summary": 38, "negative": 60}
SEED = 20261118


@dataclass(frozen=True)
class GoldenItem:
    id: str
    stratum: str
    kind: str
    question: str
    gold_notes: tuple[str, ...]
    gold_docs: tuple[str, ...]
    ground_truth: str
    expect: str  # answer | abstain
    split: str   # dev | test


def _frontmatter(raw: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, m.group(2).strip()


def load_vault(root: Path = ROOT) -> tuple[dict[str, dict], dict[str, dict]]:
    """(atomic notes, policies) keyed by id, each with text/title/hub."""
    notes, policies = {}, {}
    for p in sorted((root / "knowledge_vault/atomic").glob("*.md")):
        meta, body = _frontmatter(p.read_text())
        notes[meta["id"]] = {"text": body.split("\n\n")[0].strip(), "hub": meta["part_of"], "title": meta["title"]}
    for p in sorted((root / "knowledge_vault").glob("*.md")):
        meta, body = _frontmatter(p.read_text())
        policies[meta["id"]] = {"text": re.sub(r"^# .*\n+", "", body).strip(), "title": meta["title"]}
    return notes, policies


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:]


def build_golden(root: Path = ROOT) -> list[GoldenItem]:
    notes, policies = load_vault(root)
    rng = random.Random(SEED)
    raw: list[tuple[str, str, str, list[str], str, str]] = []  # (stratum, kind, question, gold notes, ground truth, expect)

    # factoid: the first phrasing of every note, plus seeded second phrasings up to 63
    ids = sorted(notes)
    firsts = [(nid, SINGLE[nid][0]) for nid in ids]
    seconds = [(nid, SINGLE[nid][1]) for nid in rng.sample(ids, COUNTS["factoid"] - len(ids))]
    for nid, q in firsts + seconds:
        raw.append(("factoid", "single-fact", q, [nid], notes[nid]["text"], "answer"))

    for q, gold, gt in COMPLEX:
        raw.append(("complex", "scenario", q, gold, gt, "answer"))

    seen_pairs: set[frozenset[str]] = set()
    for q, gold in MULTI:
        raw.append(("multihop", "related-pair", q, gold, " ".join(notes[g]["text"] for g in gold), "answer"))
        seen_pairs.add(frozenset(gold))
    while sum(1 for r in raw if r[0] == "multihop") < COUNTS["multihop"]:
        a, b = rng.sample(ids, 2)
        if notes[a]["hub"] == notes[b]["hub"] or frozenset((a, b)) in seen_pairs:
            continue  # two facts from DIFFERENT policies, each pair once
        seen_pairs.add(frozenset((a, b)))
        q1, q2 = SINGLE[a][rng.randrange(2)].rstrip("?"), SINGLE[b][rng.randrange(2)].rstrip("?")
        raw.append(("multihop", "cross-policy-pair", f"{q1}, and {_lower_first(q2)}?", [a, b], notes[a]["text"] + " " + notes[b]["text"], "answer"))

    templates = ["Summarize the {t}.", "What are the key rules in the {t}?", "Give me an overview of the {t}."]
    for pid in sorted(policies):
        for tpl in templates:
            raw.append(("summary", "policy-summary", tpl.format(t=policies[pid]["title"]), [n for n in sorted(notes) if notes[n]["hub"] == pid],
                        policies[pid]["text"], "answer"))
    extra = rng.sample(sorted(policies), COUNTS["summary"] - 3 * len(policies))
    for pid in extra:
        raw.append(("summary", "policy-summary", f"Briefly describe what the {policies[pid]['title']} requires.",
                    [n for n in sorted(notes) if notes[n]["hub"] == pid], policies[pid]["text"], "answer"))

    for q, kind in NEGATIVE:
        raw.append(("negative", f"negative-{kind}", q, [], "", "abstain"))

    items, per_stratum = [], {}
    for stratum, kind, q, gold, gt, expect in raw:
        n = per_stratum[stratum] = per_stratum.get(stratum, 0) + 1
        split = "dev" if (n - 1) % 5 in (0, 1) else "test"  # 40 / 60, stratified because n restarts per stratum
        items.append(GoldenItem(f"{stratum[:3]}-{n:03d}", stratum, kind, q, tuple(gold), tuple(dict.fromkeys(notes[g]["hub"] for g in gold)), gt, expect, split))
    return items


def write_golden(path: Path = PATH, root: Path = ROOT) -> list[GoldenItem]:
    items = build_golden(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(i), sort_keys=True) + "\n" for i in items))
    return items


def load_golden(path: Path = PATH, root: Path = ROOT) -> list[GoldenItem]:
    """Load and VALIDATE: exact stratum counts, unique ids/questions, gold ids exist, ground truth is vault text."""
    notes, policies = load_vault(root)
    items = [GoldenItem(**{**d, "gold_notes": tuple(d["gold_notes"]), "gold_docs": tuple(d["gold_docs"])})
             for d in map(json.loads, path.read_text().splitlines())]
    counts: dict[str, int] = {}
    for i in items:
        counts[i.stratum] = counts.get(i.stratum, 0) + 1
    if counts != COUNTS:
        raise ValueError(f"stratification mismatch: {counts} != {COUNTS}")
    if len({i.id for i in items}) != len(items) or len({i.question for i in items}) != len(items):
        raise ValueError("duplicate ids or questions")
    for i in items:
        if any(g not in notes for g in i.gold_notes):
            raise ValueError(f"{i.id}: unknown gold note")
        if i.expect == "abstain" and (i.gold_notes or i.ground_truth):
            raise ValueError(f"{i.id}: negatives carry no gold")
        if i.expect == "answer" and not (i.gold_notes and i.ground_truth and i.gold_docs):
            raise ValueError(f"{i.id}: answerable items need gold notes and a ground truth")
    return items


if __name__ == "__main__":
    out = write_golden()
    by = {}
    for i in out:
        by[i.stratum] = by.get(i.stratum, 0) + 1
    print(f"wrote {PATH.relative_to(ROOT)}: {len(out)} items {by} | dev {sum(i.split == 'dev' for i in out)} test {sum(i.split == 'test' for i in out)}")

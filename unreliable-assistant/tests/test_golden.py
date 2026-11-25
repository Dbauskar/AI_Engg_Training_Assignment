import json
from collections import Counter

import pytest

from evals.golden import COUNTS, PATH, build_golden, load_golden, load_vault, write_golden


@pytest.fixture(scope="module")
def items():
    return load_golden()


def test_stratification_matches_the_target_proportions(items):
    c = Counter(i.stratum for i in items)
    assert dict(c) == COUNTS and sum(c.values()) == 273
    # design v1.9: negatives raised from 15% to ~22% (more abstention evidence); the other strata keep their counts
    for stratum, target in (("factoid", 63), ("complex", 62), ("multihop", 50), ("summary", 38), ("negative", 60)):
        assert c[stratum] == target, stratum


def test_ids_and_questions_are_unique_and_the_split_is_stratified(items):
    assert len({i.id for i in items}) == 273 and len({i.question for i in items}) == 273
    for stratum in COUNTS:
        dev = sum(i.split == "dev" for i in items if i.stratum == stratum)
        n = COUNTS[stratum]
        assert abs(dev / n - 0.4) < 0.05, f"{stratum}: {dev}/{n} in dev"
    assert {i.split for i in items} == {"dev", "test"}


def test_every_gold_reference_exists_and_ground_truth_comes_from_the_vault(items):
    notes, policies = load_vault()
    for i in items:
        if i.expect == "abstain":
            assert i.stratum == "negative" and not i.gold_notes
            continue
        assert i.gold_notes and all(g in notes for g in i.gold_notes)
        assert set(i.gold_docs) == {notes[g]["hub"] for g in i.gold_notes}
        if i.stratum in ("factoid", "multihop"):
            assert all(notes[g]["text"] in i.ground_truth for g in i.gold_notes)  # ground truth is the vault text, verbatim
        if i.stratum == "summary":
            assert i.ground_truth == policies[i.gold_docs[0]]["text"] and len(i.gold_notes) >= 3


def test_multihop_questions_need_two_different_policies_and_complex_ones_are_scenarios(items):
    for i in items:
        if i.kind == "cross-policy-pair":
            assert len(i.gold_notes) == 2 and len(i.gold_docs) == 2
        if i.stratum == "multihop":
            assert len(i.gold_notes) >= 2
    assert Counter(i.kind for i in items if i.stratum == "negative") == {"negative-oos": 15, "negative-uncovered": 45}


def test_generation_is_deterministic_and_the_committed_file_is_current(items, tmp_path):
    again = write_golden(tmp_path / "g.jsonl")
    assert (tmp_path / "g.jsonl").read_bytes() == PATH.read_bytes(), "run `uv run python -m evals.golden` after editing the sources"
    assert [i.id for i in again] == [i.id for i in items]


def test_loader_rejects_a_tampered_dataset(tmp_path):
    lines = PATH.read_text().splitlines()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(lines[:-1]) + "\n")  # one item short
    with pytest.raises(ValueError, match="stratification"):
        load_golden(bad)
    d = json.loads(lines[0])
    d["gold_notes"] = ["no-such-note"]
    bad.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="unknown gold note"):
        load_golden(bad)
    dup = json.loads(lines[1])
    dup["question"] = json.loads(lines[0])["question"]
    bad.write_text("\n".join([lines[0], json.dumps(dup)] + lines[2:]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(bad)


def test_judge_verbatim_quote_rule_accepts_copied_spans_only():
    from evals.run_ragas import verbatim_quote_supported as v
    ctx = "Damage caused by liquid ingress or unauthorised repair voids the warranty."
    assert v('The policy states that "damage caused by liquid ingress or unauthorised repair voids the warranty."', ctx)
    assert not v('The policy states that "damage caused by liquid ingress never voids the warranty at all."', ctx)  # altered span
    assert not v('Everyone agrees, after a long discussion of many things, that "damage caused by liquid ingress or unauthorised repair voids the warranty."', ctx)  # long wrapper

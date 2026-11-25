import math

import pytest

from evals.metrics import (context_entity_recall, context_precision, context_recall, extract_entities, mean, mrr_at_k,
                           ndcg_at_k, recall_at_k)


def test_mrr_is_the_reciprocal_rank_of_the_first_relevant_item():
    assert mrr_at_k(["a", "b", "c"], {"b"}) == 0.5
    assert mrr_at_k(["a", "b", "c"], {"a", "c"}) == 1.0
    assert mrr_at_k(["a", "b", "c", "d", "e"], {"e"}, k=4) == 0.0  # outside the cut-off
    assert mrr_at_k([], {"a"}) == 0.0


def test_ndcg_matches_a_hand_computed_value():
    # relevant at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.5 ; ideal = 1 + 1/log2(3)
    assert ndcg_at_k(["a", "b", "c", "d"], {"a", "c"}) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "c", "x", "y"], {"a", "c"}) == pytest.approx(1.0)  # perfect ordering
    assert ndcg_at_k(["x", "y"], {"a"}) == 0.0
    assert ndcg_at_k(["a"], set()) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 1) == 0.5
    assert recall_at_k(["a"], set(), 5) == 1.0  # nothing was required


def test_context_precision_is_ragas_average_precision():
    # relevant at ranks 1 and 3: precision@1 = 1, precision@3 = 2/3 -> (1 + 2/3) / 2
    assert context_precision(["a", "b", "c", "d"], {"a", "c"}) == pytest.approx(5 / 6)
    assert context_precision(["a", "c", "b", "d"], {"a", "c"}) == 1.0
    assert context_precision(["x", "y"], {"a"}) == 0.0
    assert context_precision(["x", "a"], {"a"}) == 0.5  # a relevant chunk ranked second is worth half


def test_context_recall_counts_gold_sentences_present_in_the_context():
    ctx = "Refunds above 5,000 USD need approval. Customers may request a refund within 30 days of purchase."
    assert context_recall(ctx, ["Customers may request a refund within 30 days of purchase."]) == 1.0
    assert context_recall(ctx, ["Customers may request a refund within 30 days of purchase.", "Alcohol is not reimbursable."]) == 0.5
    assert context_recall("", ["Anything."]) == 0.0
    assert context_recall("x", []) == 1.0
    assert context_recall("refunds above 5000 usd need approval", ["Refunds above 5,000 USD need approval."]) == 1.0  # commas/case are normalised


def test_entity_extraction():
    e = extract_entities("The enterprise licence SKU-099 is non-refundable. HR and Tax approve more than 30 days; freeze runs 15 December to 3 January. Tier 1 vendors hold a SOC 2 Type II report.")
    assert {"sku-099", "30", "15", "3", "december", "january", "hr", "soc", "2", "1"} <= e
    assert "the" not in e and "the enterprise licence" not in e  # sentence-initial words are not entities
    assert "5000" in extract_entities("Refunds above 5,000 USD")  # thousands separators are stripped


def test_context_entity_recall():
    ref = "Wires above 10,000 USD need two approvers; above 250,000 USD the compliance officer also signs."
    assert context_entity_recall(ref, ref) == 1.0
    # entities of ref = {10000, 250000, usd}; the context holds 10000 and usd but not 250000
    assert extract_entities(ref) == {"10000", "250000", "usd"}
    assert context_entity_recall(ref, "Wires above 10,000 USD need two approvers.") == pytest.approx(2 / 3)
    assert context_entity_recall("no entities here at all", "whatever") == 1.0
    assert context_entity_recall("Limit is 75 USD per day", "unrelated text") == 0.0


def test_mean_handles_empty_input():
    assert mean([1, 2, 3]) == 2 and math.isnan(mean([]))

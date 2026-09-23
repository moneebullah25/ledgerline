import math

from ledgerline.evals.retrieval import _value_pattern, ndcg10, recall_capped, renderings


def test_renderings_match_printed_values_only():
    pat = _value_pattern(renderings(94_930_000_000, "USD"))
    assert pat.search("Total net sales | $94,930 | 85,777")
    assert pat.search("revenue of $94.9 billion")
    assert not pat.search("194,930") and not pat.search("94,930.5")
    assert pat.search("Net loss | (94,930)")
    assert renderings(1.57, "USD/share") == ["1.57"]
    assert renderings(4_000_000, "USD") == ["4,000"]  # "4" alone would match everything


def test_ndcg_and_capped_recall():
    assert ndcg10([1, 2, 3], {1}) == 1.0
    assert math.isclose(ndcg10([9, 1], {1}), 1 / math.log2(3))
    assert recall_capped([1, 9], {1, 2}) == 0.5

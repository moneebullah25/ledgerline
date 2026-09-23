from datetime import date

import pytest

from ledgerline.graph.agents.xbrl_table import assess, mentions_value, periods, printed_half_ulp
from ledgerline.models import Claim
from ledgerline.store.xbrl_query import Fact


def C(text, value, unit="USD", metric="revenue", **kw) -> Claim:
    base = {"company_id": 1, "fiscal_year": 2025, "fiscal_period": "Q1", "span_start": 0,
            "span_end": 1, "claim_type": "reported_result", "filing_id": 1,
            "target_fiscal_year": 2025, "target_fiscal_period": "Q1"}
    return Claim(**(base | {"text": text, "value": value, "unit": unit, "metric": metric} | kw))


def F(value, fy=2025, fp="Q1", concept="revenue", ids=(11,), derived=False) -> Fact:
    return Fact(value, "USD", ids, concept, fy, fp, date(2025, 3, 29), derived, "u")


def test_half_ulp_follows_printed_digits():
    assert printed_half_ulp("revenue of $94.9 billion", 94.9e9) == pytest.approx(0.05e9)
    assert printed_half_ulp("revenue of $95 billion", 95e9) == pytest.approx(0.5e9)
    assert printed_half_ulp("up 6.07%", 6.07) == pytest.approx(0.005)
    assert printed_half_ulp("no number here", 5.0) == 0.0


def test_rounding_aware_level():
    claim = C("revenue of $94.9 billion", 94.9e9)
    ok = assess(claim, [F(94_930e6)])
    assert ok.stance == "supports"
    assert ok.citations[0].ref_id == 11 and ok.citations[0].kind == "xbrl_fact"
    bad = assess(claim, [F(97_000e6)])
    assert bad.stance == "contradicts"
    assert bad.payload["deviation"] == pytest.approx((94.9e9 - 97e9) / 97e9)


def test_range_claim_inside_and_outside():
    claim = C("between $89 billion and $93 billion", 89e9, value_high=93e9)
    assert assess(claim, [F(91e9)]).stance == "supports"
    assert assess(claim, [F(95e9)]).stance == "contradicts"


def test_unit_mismatch_is_insufficient():
    assert assess(C("up 5%", 5.0, unit="percent"), [F(94e9)]).stance == "insufficient"


def test_missing_fact_is_insufficient_not_a_guess():
    ev = assess(C("revenue of $94.9 billion", 94.9e9), [None])
    assert ev.stance == "insufficient" and ev.citations == []


def test_comparison_percent_change():
    claim = C("revenue grew 6% year over year", 6.0, unit="percent", claim_type="comparison",
              comparison="yoy")
    base, cur = F(100e9, fy=2024, ids=(1,)), F(106.3e9, ids=(2,))
    ev = assess(claim, [cur, base])
    assert ev.stance == "supports"  # 6.3% rounds to "6%" within 0.5 + 0.1 pp
    assert {c.ref_id for c in ev.citations} == {1, 2}
    assert assess(claim, [F(107e9, ids=(2,)), base]).stance == "contradicts"  # 7.0%


def test_margin_change_is_in_points():
    claim = C("gross margin up 1.2 points", 1.2, unit="percent", metric="gross_margin",
              claim_type="comparison", comparison="yoy")
    ev = assess(claim, [F(46.3, concept="gross_margin"), F(45.1, fy=2024, concept="gross_margin")])
    assert ev.stance == "supports"


def test_comparison_periods():
    assert periods(C("x 5%", 5.0, comparison="qoq", target_fiscal_period="Q1")) == [
        (2025, "Q1"), (2024, "Q4")]
    assert periods(C("x 5%", 5.0, comparison="yoy", target_fiscal_period="Q3")) == [
        (2025, "Q3"), (2024, "Q3")]
    assert periods(C("x 5%", 5.0, comparison="qoq", target_fiscal_period="FY")) is None


def test_mentions_value_any_scale():
    claim = C("revenue of $94.9 billion", 94.9e9)
    assert mentions_value("Total net sales were $94,930 million.", claim)
    assert not mentions_value("Total net sales were $97,000 million.", claim)
    pct = C("revenue up 18% sequentially", 18.0, unit="percent", claim_type="comparison",
            comparison="qoq")
    assert mentions_value("Revenue increased 18% from the prior quarter.", pct)
    assert not mentions_value("repurchases in compliance with Rule 10b-18.", pct)
    assert not mentions_value("an unfavorable impact of 5.9 million units", pct)

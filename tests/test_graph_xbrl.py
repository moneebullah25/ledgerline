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


def F(value, fy=2025, fp="Q1", concept="revenue", ids=(11,), derived=False, unit="USD",
      components=()) -> Fact:
    return Fact(value, unit, ids, concept, fy, fp, date(2025, 3, 29), derived, "u", components)


def margin(gp, rev, fy=2025, ids=(1, 2)) -> Fact:
    parts = (F(gp, fy, concept="gross_profit", ids=ids[:1]), F(rev, fy, ids=ids[1:]))
    return F(gp / rev * 100, fy, concept="gross_margin", ids=ids, unit="percent", components=parts)


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
    assert {c.ref_id for c in ev.citations if c.kind == "xbrl_fact"} == {1, 2}
    change = next(c for c in ev.citations if c.ref_id == "revenue:2025Q1:yoy_change")
    assert (change.kind, change.unit) == ("computed", "percent")
    assert change.value == pytest.approx(6.3)
    assert assess(claim, [F(107e9, ids=(2,)), base]).stance == "contradicts"  # 7.0%


def test_margin_change_is_in_points():
    claim = C("gross margin up 1.2 points", 1.2, unit="percent", metric="gross_margin",
              claim_type="comparison", comparison="yoy")
    ev = assess(claim, [margin(46.3, 100), margin(45.1, 100, fy=2024, ids=(3, 4))])
    assert ev.stance == "supports"


def test_margin_cites_components_with_their_own_values():
    claim = C("gross margin of 46.3%", 46.3, unit="percent", metric="gross_margin")
    ev = assess(claim, [margin(43.9e9, 94.9e9)])
    facts = {c.ref_id: (c.value, c.unit) for c in ev.citations if c.kind == "xbrl_fact"}
    assert facts == {1: (43.9e9, "USD"), 2: (94.9e9, "USD")}
    ratio = next(c for c in ev.citations if c.ref_id == "gross_margin:2025Q1")
    assert ratio.kind == "computed" and ratio.value == pytest.approx(46.26, abs=0.01)
    assert not ev.payload["derived"]  # computed, not derived from YTD


def test_percent_level_uses_points_not_relative_tolerance():
    claim = C("gross margin of 46%", 46.0, unit="percent", metric="gross_margin")
    assert assess(claim, [margin(46.55, 100)]).stance == "supports"  # 0.55 <= 0.5 + 0.1
    assert assess(claim, [margin(46.65, 100)]).stance == "contradicts"  # was ok at 0.5% rel


def test_usd_change_tolerance_scales_with_both_values():
    claim = C("revenue increased $1.0 billion", 1.0e9, claim_type="comparison", comparison="yoy")
    ev = assess(claim, [F(101.3e9), F(100e9, fy=2024, ids=(1,))])  # actual change 1.3e9
    assert ev.payload["tolerance"] == pytest.approx(0.005 * 201.3e9 + 0.05e9)
    assert ev.stance == "supports"


def test_comparison_unit_must_be_percent_or_concept_unit():
    claim = C("revenue up 5 per share", 5.0, unit="USD/share", claim_type="comparison",
              comparison="yoy")
    assert assess(claim, [F(105e9), F(100e9, fy=2024)]).stance == "insufficient"


def test_half_ulp_prefers_percent_tokens_for_percent_claims():
    text = "revenue of $5.0 billion, up 5%"
    assert printed_half_ulp(text, 5.0) == pytest.approx(0.05)
    assert printed_half_ulp(text, 5.0, "percent") == pytest.approx(0.5)


def test_net_loss_claim_is_compared_with_negative_fact():
    loss = C("net loss of $2.1 billion", 2.1e9, metric="net_income")
    assert assess(loss, [F(-2.1e9, concept="net_income")]).stance == "supports"
    assert assess(loss, [F(2.1e9, concept="net_income")]).stance == "contradicts"
    income = C("net income of $2.1 billion", 2.1e9, metric="net_income")
    assert assess(income, [F(-2.1e9, concept="net_income")]).stance == "contradicts"
    op = C("operating loss of $300 million", 300e6, metric="operating_income")
    assert assess(op, [F(-301e6, concept="operating_income")]).stance == "supports"


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
    fell = C("revenue fell 5% year over year", -5.0, unit="percent", claim_type="comparison",
             comparison="yoy")
    assert mentions_value("Revenue decreased 5% compared with last year.", fell)


def test_flat_is_a_whole_percent_zero():
    claim = C("revenue was $12.7 billion, flat year-over-year", 0.0, unit="percent",
              claim_type="comparison", comparison="yoy")
    assert printed_half_ulp(claim.text, 0.0, "percent") == 0.5
    assert assess(claim, [F(12_667e6), F(12_724e6, fy=2024)]).stance == "supports"   # -0.4%
    assert assess(claim, [F(12_630e6), F(12_724e6, fy=2024)]).stance == "contradicts"  # -0.7%


@pytest.mark.parametrize("text,value", [
    ("EPS includes an expected gain from the exit of the Glad Joint Venture in the range of $0.10",
     0.10),
    (("EPS includes a gain of $0.10 to $0.13 per share and restructuring charges of $0.12 to "
      "$0.25 per share"), 0.25),
    ("$0.30 of diluted earnings per share", 0.30),
])
def test_component_of_eps_is_insufficient_not_compared_with_total(text, value):
    claim = C(text, value, unit="USD/share", metric="eps_diluted")
    ev = assess(claim, [F(6.62, unit="USD/share", concept="eps_diluted")])
    assert ev.stance == "insufficient" and "component" in ev.summary


def test_total_with_an_included_part_is_still_compared():
    claim = C("Diluted EPS, including a charge of $0.10, was $2.00", 2.0, unit="USD/share",
              metric="eps_diluted")
    assert assess(claim, [F(2.0, unit="USD/share", concept="eps_diluted")]).stance == "supports"
    assert assess(C("diluted EPS of $0.30", 0.3, unit="USD/share", metric="eps_diluted"),
                  [F(6.62, unit="USD/share", concept="eps_diluted")]).stance == "contradicts"


def test_company_defined_free_cash_flow_mismatch_is_insufficient():
    parts = (F(27_988e6, concept="operating_cash_flow", ids=(1,)),
             F(14_425e6, concept="capex", ids=(2,)))
    fcf = F(13_563e6, concept="free_cash_flow", ids=(1, 2), components=parts)
    claim = C("Free cash flow was $13.15 billion", 13.15e9, metric="free_cash_flow")
    ev = assess(claim, [fcf])
    assert ev.stance == "insufficient" and ev.payload["actual"] == 13_563e6
    assert assess(C("Free cash flow was $13.56 billion", 13.56e9, metric="free_cash_flow"),
                  [fcf]).stance == "supports"

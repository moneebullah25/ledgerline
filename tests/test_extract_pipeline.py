from pathlib import Path

import pytest

from ledgerline.extract.pipeline import extract_candidates, extract_claims

FIX = Path(__file__).parent / "fixtures"


def claims(text, fy=2026, fp="Q3", **kw):
    return extract_claims(text, fiscal_year=fy, fiscal_period=fp, use_ner=False, **kw)


def one(text, **kw):
    (c,) = claims(text, **kw)
    return c


@pytest.mark.parametrize("name,fy,fp,fye", [
    ("nvda_q2fy27", 2027, "Q2", "0131"), ("nvda_q1fy27", 2027, "Q1", "0131"),
    ("aapl_q3fy26", 2026, "Q3", "0926"), ("aapl_q2fy26", 2026, "Q2", "0926"),
    ("msft_q4fy26", 2026, "Q4", "0630"),
])
def test_spans_index_the_source_exactly(name, fy, fp, fye):
    text = (FIX / f"ex991_{name}.txt").read_text()
    cs = claims(text, fy, fp, fiscal_year_end=fye)
    assert cs
    for c in cs:
        assert text[c.span_start:c.span_end] == c.text
    assert len({(c.span_start, c.span_end) for c in cs}) == len(cs)


def test_nvda_release_key_claims():
    text = (FIX / "ex991_nvda_q2fy27.txt").read_text()
    cs = claims(text, 2027, "Q2", fiscal_year_end="0131")
    got = {(c.claim_type, c.metric, c.value, c.value_high, c.target_fiscal_year, c.target_fiscal_period,
            c.comparison) for c in cs}
    assert ("reported_result", "revenue", 96.2e9, None, 2027, "Q2", None) in got
    assert ("comparison", "revenue", 106.0, None, 2027, "Q2", "yoy") in got
    assert ("comparison", "revenue", 18.0, None, 2027, "Q2", "qoq") in got
    # outlook section: the lead-in line sets the target quarter for the bullets under it
    assert ("guidance", "revenue", 105.84e9, 110.16e9, 2027, "Q3", None) in got
    assert ("guidance", "gross_margin", 73.5, 74.5, 2027, "Q3", None) in got
    # table row, "$ in millions" header
    assert ("reported_result", "net_income", 59688e6, None, 2027, "Q2", None) in got
    # Data Center is a segment: never mapped to company revenue
    assert not any(c.metric == "revenue" and c.value == 89.0e9 for c in cs)
    # "GAAP and non-GAAP ... $2.46 and $2.22, respectively": the non-GAAP figure is dropped
    assert not any(c.value == 2.22 for c in cs)
    # the non-GAAP summary table is skipped entirely
    assert not any(c.value == 53954e6 for c in cs)


def test_msft_fiscal_year_section_targets_fy():
    text = (FIX / "ex991_msft_q4fy26.txt").read_text()
    cs = claims(text, 2026, "Q4", fiscal_year_end="0630")
    rev = {(c.value, c.target_fiscal_period) for c in cs if c.metric == "revenue" and c.claim_type == "reported_result"}
    assert (90.0e9, "Q4") in rev and (331.8e9, "FY") in rev
    assert not any(c.value == 35.3e9 for c in cs)  # "... on a non-GAAP basis"
    assert not any(c.value == 37847e6 for c in cs)  # segment results table


def test_aapl_table_scale_and_gross_margin_dollars():
    text = (FIX / "ex991_aapl_q3fy26.txt").read_text()
    cs = claims(text, 2026, "Q3", fiscal_year_end="0926")
    got = {(c.metric, c.value, c.unit) for c in cs}
    assert ("revenue", 109417e6, "USD") in got
    assert ("gross_profit", 54770e6, "USD") in got  # Apple's "Gross margin" row is dollars
    assert ("gross_margin", 50.1, "percent") in got
    # "including a favorable impact of approximately 2 percentage points" is a component
    assert not any(c.metric == "gross_margin" and c.value == 2.0 for c in cs)


def facts(text, **kw):
    return [(c.claim_type, c.metric, c.value, c.value_high, c.target_fiscal_year, c.target_fiscal_period,
             c.comparison) for c in claims(text, **kw)]


def test_reported_and_comparison_from_one_sentence():
    assert facts("Revenue grew 8% year over year to $94.9 billion.") == [
        ("comparison", "revenue", 8.0, None, 2026, "Q3", "yoy"),
        ("reported_result", "revenue", 94.9e9, None, 2026, "Q3", None),
    ]


def test_negative_change_and_qoq():
    assert facts("Gross margin was 46.5 percent, down 30 basis points sequentially.") == [
        ("reported_result", "gross_margin", 46.5, None, 2026, "Q3", None),
        ("comparison", "gross_margin", -0.3, None, 2026, "Q3", "qoq"),
    ]


def test_guidance_targets_the_named_quarter():
    assert facts("For the fourth quarter, we expect revenue between $89 billion and $93 billion.") == [
        ("guidance", "revenue", 89e9, 93e9, 2026, "Q4", None)]
    # AAPL-style calendar-month quarter, resolved through the fiscal calendar
    assert facts("We expect revenue for the December quarter to grow 10 to 12 percent year over year.",
                 fp="Q4", fiscal_year_end="0926") == [("guidance", "revenue", 10.0, 12.0, 2027, "Q1", "yoy")]


def test_guidance_heading_carries_to_bullets():
    text = "Fiscal 2026 Guidance\n•Gross margin of approximately 33.1%"
    assert facts(text, fp="Q2") == [("guidance", "gross_margin", 33.1, None, 2026, "FY", None)]


def test_baseline_values_are_not_claims():
    assert facts("Diluted earnings per share was $1.10, compared with $0.95 in the same period last year.") == [
        ("reported_result", "eps_diluted", 1.1, None, 2026, "Q3", None)]


def test_boilerplate_and_non_claims_are_discarded():
    for text in [
        ("This press release contains forward-looking statements within the meaning of the Private "
         "Securities Litigation Reform Act of 1995. Revenue was $5 billion."),
        "The Company assumes no obligation to update any forward-looking statements.",
        "One Infinite Loop, Cupertino, CA 95014",
        "July 30, 2026",
        "NVIDIA will conduct a conference call with analysts at 2 p.m. Pacific time.",
    ]:
        assert claims(text) == [], text


def test_qualitative_claim():
    (c,) = claims("Our record revenue and strong margins reflect robust demand across the business.")
    assert (c.claim_type, c.metric, c.value) == ("qualitative", "revenue", None)


def test_unmapped_metric_becomes_qualitative_with_value():
    (cand,) = extract_candidates("Data Center revenue was $89.0 billion.", fiscal_year=2027,
                                 fiscal_period="Q2", use_ner=False)
    assert (cand.claim_type, cand.metric) == ("reported_result", None)  # what typing decided
    (c,) = claims("Data Center revenue was $89.0 billion.", fy=2027, fp="Q2")
    assert (c.claim_type, c.metric, c.value, c.unit) == ("qualitative", None, 89.0e9, "USD")


def test_table_needs_scale_and_period_header():
    rows = "June 27, 2026 | June 28, 2025\nNet sales | 94,930 | 85,777\nNet income | (1,234) | 500"
    assert claims("Three Months Ended\n" + rows) == []  # no "in millions": skip, do not guess
    assert claims("(In millions)\n" + rows) == []  # no period header: skip
    assert facts("(In millions)\nThree Months Ended\n" + rows) == [
        ("reported_result", "revenue", 94930e6, None, 2026, "Q3", None),
        ("reported_result", "net_income", -1234e6, None, 2026, "Q3", None),
    ]
    # oldest column first: the first number is not the current quarter
    assert claims("(In millions)\nThree Months Ended\nJune 28, 2025 | June 27, 2026\nNet sales | 1 | 2") == []


def test_speaker_roles_drop_analysts_and_tag_executives():
    text = "Revenue was $10.0 billion, up 5% year over year.\nSo revenue was $10.0 billion?"
    split = text.index("\n") + 1
    cs = claims(text, speaker_roles=[(0, split, "executive"), (split, len(text), "analyst")])
    assert cs and all(c.speaker_role == "executive" and c.span_end <= split for c in cs)


def test_hedged_guidance_scores_higher_than_actuals():
    text = (FIX / "ex991_nvda_q2fy27.txt").read_text()
    cs = claims(text, 2027, "Q2")
    g = [c.hedging_score for c in cs if c.claim_type == "guidance"]
    r = [c.hedging_score for c in cs if c.claim_type == "reported_result"]
    assert g and min(g) > max(r)


def test_blank_lines_and_carried_metric():
    assert claims("\n\n") == []
    text = "Q4 FY 2026 Results:\n◦Revenue: $17.3 billion\n▪Increase of 18% year over year"
    got = [(c.claim_type, c.metric, c.value, c.comparison) for c in claims(text, fp="Q4")]
    assert got == [("reported_result", "revenue", 17.3e9, None), ("comparison", "revenue", 18.0, "yoy")]

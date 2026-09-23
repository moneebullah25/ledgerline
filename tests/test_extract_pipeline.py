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


# ---------------------------------------------------------------- review group E


def test_non_earnings_release_is_not_a_release():
    from ledgerline.extract.pipeline import is_earnings_release

    cfo = ("NIKE, Inc. Announces Planned CFO Transition\nBEAVERTON, Ore. — NIKE, Inc. today announced "
           "that David M. Denton will join the company as Chief Financial Officer. NIKE, Inc. will report "
           "fourth quarter and fiscal year 2026 results on Tuesday, June 30th.")
    assert not is_earnings_release(cfo)
    assert is_earnings_release((FIX / "ex991_nvda_q2fy27.txt").read_text())


@pytest.mark.parametrize("text,fp", [
    # headline under a "Raises ... Targets" heading: a present-tense result, and a sum with RPO
    (("Adobe Raises FY25 Digital Media ARR Growth, Total Revenue, and EPS Targets\n"
      "Company Reports Record Revenue and RPO Surpasses $20 Billion"), "Q3"),
    # "annual" names an event; the savings belong to a new clause
    (("Held its annual Prime Big Deal Days shopping event, with record sales, number of items sold, and "
      "Prime member participation, saving Prime members more than $1 billion with deals."), "Q3"),
    ("Total sales include approximately $900 million from the recent acquisition of GMS Inc.", "Q3"),
    ("GAAP EPS includes expected non-core restructuring charges of $0.13 to $0.17 per share.", "Q4"),
    ("53rd week expected to contribute approximately $0.30 of diluted earnings per share.", "Q3"),
])
def test_guidance_false_positives(text, fp):
    assert not [c for c in claims(text, fp=fp) if c.claim_type == "guidance" or c.metric == "revenue"
                and c.value in (20e9, 1e9, 900e6)]


def test_segment_revenue_under_company_guidance_is_demoted():
    text = ("Raising expected FY26 revenue to $4 billion, up from $3 billion\n"
            "FY 2026 Guidance:\nRevenue: $62.8 billion to $63.0 billion")
    got = {(c.metric, c.value) for c in claims(text, fp="Q3")}
    assert ("revenue", 62.8e9) in got and (None, 4e9) in got


def test_respectively_assigns_quarter_and_full_year():
    text = ("Revenue was $59.89 billion and $200.97 billion, representing increases of 24% and 22% "
            "year-over-year for the fourth quarter and full year 2025, respectively.")
    assert facts(text, fy=2025, fp="Q4") == [
        ("reported_result", "revenue", 59.89e9, None, 2025, "Q4", None),
        ("reported_result", "revenue", 200.97e9, None, 2025, "FY", None),
        ("comparison", "revenue", 24.0, None, 2025, "Q4", "yoy"),
        ("comparison", "revenue", 22.0, None, 2025, "FY", "yoy"),
    ]


def test_full_year_context_and_labels():
    # "For the quarter" wins over a heading that names the full year
    text = "Quarterly/Full Year Performance\nFor the quarter, net revenues grew 2% to $11.8 billion."
    assert ("reported_result", "revenue", 11.8e9, None, 2025, "Q4", None) in facts(text, fy=2025, fp="Q4")
    # a line label dates every clause after it
    assert facts("FY ’25: Net Sales +1%; Diluted EPS +8%", fy=2025, fp="Q4") == [
        ("comparison", "revenue", 1.0, None, 2025, "FY", "yoy"),
        ("comparison", "eps_diluted", 8.0, None, 2025, "FY", "yoy")]
    # a Q4 release's "increased to $X in 2024" is the fiscal year's level, not a dollar delta
    assert facts("Operating income increased to $68.6 billion in 2024, compared with $36.9 billion in 2023.",
                 fy=2024, fp="Q4") == [("reported_result", "operating_income", 68.6e9, None, 2024, "FY", None)]


def test_periods_we_do_not_model_are_dropped():
    assert claims("Operating cash flow increased 57% to $112.7 billion for the trailing twelve months.") == []
    assert [c for c in claims("Year-to-Date Results Discussion\nOperating profit increased 65%.")
            if c.value is not None] == []
    # "quarter and year-to-date, respectively" from the heading: only the quarter's value stays
    text = "Second-Quarter and Year-to-Date Results\nEarnings per share (EPS) increased 137% and 72%, respectively"
    assert facts(text, fp="Q2") == [("comparison", "eps_diluted", 137.0, None, 2026, "Q2", "yoy")]


def test_segment_qualifiers():
    assert [c.metric for c in claims("Revenues for Converse were $244 million, down 32 percent.")] == [None, None]
    table = ("Three Months Ended\n(In Millions; Unaudited) | Jun 27, 2026\n"
             "CCPG | DCAI | Intersegment Eliminations | Total Consolidated\nRevenue | $7,871 | $3,939 | $(4,421) | $16,128")
    assert claims(table, fp="Q2") == []


def test_real_claims_not_discarded():
    assert facts("Third Quarter 2026 Guidance\n"
                 "Net sales are expected to be between $197.0 billion and $202.0 billion, or to grow between "
                 "9% and 12% compared with third quarter 2025.", fp="Q2") == [
        ("guidance", "revenue", 197e9, 202e9, 2026, "Q3", None),
        ("guidance", "revenue", 9.0, 12.0, 2026, "Q3", "yoy")]
    assert facts("Diluted net earnings per share increased by six percent versus the prior year to $1.63.") == [
        ("comparison", "eps_diluted", 6.0, None, 2026, "Q3", "yoy"),
        ("reported_result", "eps_diluted", 1.63, None, 2026, "Q3", None)]
    assert facts("Net sales were $19.8 billion, a two percent decrease versus the prior year.") == [
        ("reported_result", "revenue", 19.8e9, None, 2026, "Q3", None),
        ("comparison", "revenue", -2.0, None, 2026, "Q3", "yoy")]


def test_guidance_tables():
    adobe = ("Financial Targets\nThe following table summarizes Adobe's fourth quarter fiscal year 2025 targets:\n"
             "Total revenue | $6.075 billion to $6.125 billion\n"
             "Digital Media segment revenue | $4.53 billion to $4.56 billion\n"
             "Earnings per share | GAAP: $4.27 to $4.32 | Non-GAAP: $5.35 to $5.40")
    assert facts(adobe, fy=2025, fp="Q3") == [
        ("guidance", "revenue", 6.075e9, 6.125e9, 2025, "Q4", None),
        ("guidance", "eps_diluted", 4.27, 4.32, 2025, "Q4", None)]
    intel = ("Business Outlook\nIntel's guidance for the third quarter of 2026 includes both GAAP and non-GAAP "
             "estimates as follows:\nQ3 2026 | GAAP | Non-GAAP\nRevenue | $15.8-16.8 billion\n"
             "Gross margin | 41.0% | 42.0%\nEarnings (Loss) Per Share Attributable to Intel—Diluted | $0.31 | $0.38")
    assert facts(intel, fp="Q2", fiscal_year_end="1227") == [
        ("guidance", "revenue", 15.8e9, 16.8e9, 2026, "Q3", None),
        ("guidance", "gross_margin", 41.0, None, 2026, "Q3", None),
        ("guidance", "eps_diluted", 0.31, None, 2026, "Q3", None)]
    # a results table under a guidance heading is not guidance: its columns are past periods
    pep = ("Guidance\nSecond-Quarter | Year-to-Date\n($ in millions, except EPS) | Q2 2026 | Q2 2025\n"
           "Net revenue | $24,181 | $22,726")
    assert not [c for c in claims(pep, fp="Q2") if c.claim_type == "guidance"]


def test_component_run_and_quarters_ended_header():
    # both parts of a driver breakdown are components, not revenue comparisons
    assert not [c for c in claims("Revenue performance included 6% growth in price/mix and a 1% decline in "
                                  "concentrate sales.") if c.metric]
    # "Quarters Ended | Six Months Ended": the first column is the quarter, not the fiscal year
    table = ("Dollars in millions\nQuarters Ended June 30, | Six Months Ended June 30,\n"
             "2026 | 2025 | Inc/ (Dec) | 2026 | 2025\nRevenues | $7,099 | $6,843 | 4% | $13,616 | $12,799")
    assert facts(table, fp="Q2") == [("reported_result", "revenue", 7099e6, None, 2026, "Q2", None)]

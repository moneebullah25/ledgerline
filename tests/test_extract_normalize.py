import pytest

from ledgerline.extract.normalize import (
    find_periods,
    fiscal_quarter,
    hedging_score,
    match_metrics,
    parse_cell,
    parse_values,
    resolve_period,
)


def vals(text):
    return [(v.value, v.value_high, v.unit, v.kind) for v in parse_values(text)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Revenue was $94.9 billion", [(94.9e9, None, "USD", "money")]),
        (
            "$94.9B and $3M and $2 thousand",
            [
                (94.9e9, None, "USD", "money"),
                (3e6, None, "USD", "money"),
                (2e3, None, "USD", "money"),
            ],
        ),
        ("a loss of $(2.16)", [(-2.16, None, "USD", "money")]),
        ("-$5 million", [(-5e6, None, "USD", "money")]),
        ("EPS of $1.64 per diluted share", [(1.64, None, "USD/share", "per_share")]),
        ("up 8 percent", [(8.0, None, "percent", "percent")]),
        ("up 8%", [(8.0, None, "percent", "percent")]),
        ("expanded 890 basis points", [(8.9, None, "percent", "pp")]),
        (
            "a 4-point tailwind, up 2.6 pts",
            [(4.0, None, "percent", "pp"), (2.6, None, "percent", "pp")],
        ),
        ("between $89 billion and $93 billion", [(89e9, 93e9, "USD", "money")]),
        ("$89 billion to $93 billion", [(89e9, 93e9, "USD", "money")]),
        ("$89 to $93 billion", [(89e9, 93e9, "USD", "money")]),
        ("$108.0 billion, plus or minus 2%", [(105.84e9, 110.16e9, "USD", "money")]),
        ("74.0%, plus or minus 50 basis points", [(73.5, 74.5, "percent", "percent")]),
        ("approximately flat to 2.0%", [(0.0, 2.0, "percent", "percent")]),
        # a change then a level, not a range; two values joined by "and" outside "between"
        (
            "increased 890 basis points to 49.2 percent",
            [(8.9, None, "percent", "pp"), (49.2, None, "percent", "percent")],
        ),
        (
            "were 74.9% and 75.0%, respectively",
            [(74.9, None, "percent", "percent"), (75.0, None, "percent", "percent")],
        ),
        ("$133.7 billion", [(133.7e9, None, "USD", "money")]),
        ("in the range of $45-48 billion", [(45e9, 48e9, "USD", "money")]),
        ("EPS of $3.30-$3.40", [(3.3, 3.4, "USD", "money")]),  # exact, not 133699999999.99998
    ],
)
def test_parse_values(text, expected):
    assert vals(text) == expected


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("(1,234)", (-1234.0, None)),
        ("$96,221", (96221.0, None)),
        ("$(480)", (-480.0, None)),
        ("18%", (18.0, "%")),
        ("(4)%", (-4.0, "%")),
        ("0.1 pts", (0.1, "pts")),
        ("—", None),
        ("Q2 FY27", None),
    ],
)
def test_parse_cell(cell, expected):
    assert parse_cell(cell) == expected


def period(text, fy, fp, forward=False, fye=None):
    (_, _, kind, g), *_ = find_periods(text)
    return resolve_period(kind, g, fy, fp, forward=forward, fiscal_year_end=fye)


@pytest.mark.parametrize(
    "text,fy,fp,forward,fye,expected",
    [
        ("the third quarter of fiscal 2027", 2027, "Q2", True, None, (2027, "Q3")),
        ("its fiscal 2026 third quarter", 2026, "Q3", False, None, (2026, "Q3")),
        ("Q2 FY27", 2027, "Q2", False, None, (2027, "Q2")),
        ("Q4 FY 2026", 2026, "Q4", False, None, (2026, "Q4")),
        ("third-quarter 2026", 2026, "Q2", True, None, (2026, "Q3")),
        ("next quarter", 2026, "Q3", True, None, (2026, "Q4")),
        ("next quarter", 2026, "Q4", True, None, (2027, "Q1")),  # wraps into the next fiscal year
        ("for the first quarter", 2026, "Q4", True, None, (2027, "Q1")),  # guidance looks ahead
        ("the fourth quarter", 2026, "Q1", False, None, (2025, "Q4")),  # a Q1 release looks back
        ("full year fiscal 2027", 2027, "Q2", True, None, (2027, "FY")),
        ("for the full year", 2026, "Q2", True, None, (2026, "FY")),
        ("FY27", 2027, "Q1", True, None, (2027, "FY")),
        # fiscal calendars: AAPL's December quarter is fiscal Q1 of the next fiscal year
        ("the December quarter", 2026, "Q4", True, "0927", (2027, "Q1")),
        # MSFT (June FYE): quarter ended June 30 is Q4; NVDA (late-January FYE): April is Q1
        ("the quarter ended June 30, 2026", 2026, "Q4", False, "0630", (2026, "Q4")),
        ("the quarter ended April 26, 2026", 2027, "Q1", False, "0126", (2027, "Q1")),
        ("December quarter", 2026, "Q4", True, None, None),  # no calendar known: no guess
    ],
)
def test_resolve_period(text, fy, fp, forward, fye, expected):
    assert period(text, fy, fp, forward, fye) == expected


def test_fiscal_quarter_edge_cases():
    assert fiscal_quarter(12, 27, "0927") == 1  # AAPL
    assert fiscal_quarter(9, 27, "0927") == 4
    assert fiscal_quarter(4, 27, "0126") == 1  # NVDA
    assert fiscal_quarter(6, 30, "0630") == 4  # MSFT
    # 52/53-week calendars: a quarter or year closing on Jan 3 really closes December
    assert fiscal_quarter(1, 3, "1231") == 4
    assert fiscal_quarter(12, 28, "0103") == 4


@pytest.mark.parametrize(
    "sentence,metric",
    [
        ("Revenue of $96.2 billion", "revenue"),
        ("Total revenue was $17.3 billion", "revenue"),
        ("Second-quarter revenue was $16.1 billion", "revenue"),
        ("Cisco reported fourth quarter revenue of $17.3 billion", "revenue"),
        ("Data Center revenue of $89.0 billion", None),
        ("This year, Azure revenue surpassed $100 billion", None),
        ("Wholesale revenues were $6.6 billion", None),
        ("iPhone revenue grew", None),
        ("Revenue in Intelligent Cloud was $39.3 billion", None),
        ("Diluted earnings per share was $2.02", "eps_diluted"),
    ],
)
def test_match_metrics_qualifiers(sentence, metric):
    assert match_metrics(sentence)[0].metric == metric


def test_match_metrics_flags_non_gaap():
    (m,) = match_metrics("non-GAAP EPS attributable to Intel was $0.42")
    assert m.metric == "eps_diluted" and m.non_gaap
    (m,) = match_metrics("GAAP and non-GAAP gross margins were both 75.0%")
    assert not m.non_gaap


def test_hedging_score():
    assert hedging_score("Revenue was $5.0 billion") == 0.0
    # 2 hedges in 11 words: min(1, 2 / 2.1)
    assert hedging_score(
        "Revenue is expected to be $108.0 billion, plus or minus 2%"
    ) == pytest.approx(0.952)
    long = "Operating income was approximately $2.0 billion " + "in the period " * 8
    assert 0 < hedging_score(long) < hedging_score("approximately $2.0 billion") <= 1

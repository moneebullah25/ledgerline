from datetime import date

import pytest

from ledgerline.graph.router import route
from ledgerline.models import Claim


def C(**kw) -> Claim:
    base = {"company_id": 1, "fiscal_year": 2025, "fiscal_period": "Q1", "text": "x",
            "span_start": 0, "span_end": 1, "claim_type": "reported_result", "filing_id": 1}
    return Claim(**(base | kw))


E = date(2025, 1, 30)

CASES = [
    # (claim, event_date, reported_through, expected)
    (C(metric="revenue", value=94.9e9, unit="USD"), None, None, ["xbrl_table", "filing_rag"]),
    (C(metric="revenue", value=94.9e9, unit="USD"), E, None,
     ["xbrl_table", "filing_rag", "market_reaction"]),
    (C(metric=None, value=5, unit="percent"), None, None, ["filing_rag"]),  # unmapped metric
    (C(metric="made_up", value=5, unit="USD"), None, None, ["filing_rag"]),
    (C(metric="revenue", value=None), None, None, ["filing_rag"]),  # no number
    (C(claim_type="comparison", metric="revenue", value=6.0, unit="percent", comparison="yoy"),
     None, None, ["xbrl_table", "filing_rag"]),
    (C(claim_type="qualitative"), E, None, ["filing_rag", "market_reaction"]),
    (C(claim_type="qualitative", metric="revenue", value=1.0, unit="USD"), None, None,
     ["filing_rag"]),
    # guidance: forecast; xbrl_table only once the target period has been reported
    (C(claim_type="guidance", metric="revenue", value=89e9, value_high=93e9, unit="USD",
       target_fiscal_year=2025, target_fiscal_period="Q2"), None, (2025, "Q1"),
     ["forecast", "filing_rag"]),
    (C(claim_type="guidance", metric="revenue", value=89e9, unit="USD",
       target_fiscal_year=2025, target_fiscal_period="Q2"), None, (2025, "Q2"),
     ["forecast", "xbrl_table", "filing_rag"]),
    (C(claim_type="guidance", metric="revenue", value=89e9, unit="USD",
       target_fiscal_year=2025, target_fiscal_period="FY"), None, (2025, "Q4"),
     ["forecast", "xbrl_table", "filing_rag"]),  # FY is reported with Q4
    (C(claim_type="guidance", metric="revenue", value=89e9, unit="USD",
       target_fiscal_year=2025, target_fiscal_period="FY"), None, (2025, "Q3"),
     ["forecast", "filing_rag"]),
    (C(claim_type="guidance", metric="revenue", value=89e9, unit="USD"), None, (2026, "Q1"),
     ["forecast", "filing_rag"]),  # no target period: cannot be matched to a report
    (C(claim_type="guidance", metric=None, value=None), E, None,
     ["filing_rag", "market_reaction"]),
    (C(metric="net_income", value=0.0, unit="USD"), None, None, ["xbrl_table", "filing_rag"]),
    (C(claim_type="guidance", metric="revenue", value=89e9, unit="USD",
       target_fiscal_year=2025), None, (2026, "Q1"), ["forecast", "filing_rag"]),
    (C(claim_type="guidance", metric="revenue", value=89e9, unit="USD",
       target_fiscal_period="Q2"), None, (2026, "Q1"), ["forecast", "filing_rag"]),
]


@pytest.mark.parametrize(("claim", "event", "reported", "expected"), CASES)
def test_route(claim, event, reported, expected):
    assert route(claim, event_date=event, reported_through=reported) == expected


def test_route_is_pure_without_context():
    assert route(C(metric="revenue", value=1e9, unit="USD")) == ["xbrl_table", "filing_rag"]

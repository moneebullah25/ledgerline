"""get_xbrl_fact / get_series against a real Postgres: tag preference, duration filtering,
derived concepts, amendment dedupe, and every None path."""

from datetime import UTC, date, datetime

import pytest

from ledgerline.models import Company, XbrlFact
from ledgerline.store.xbrl_query import get_series, get_xbrl_fact

pytestmark = pytest.mark.db
CIK = "0000000042"


@pytest.fixture
def co(db):
    c = Company(cik=CIK, ticker="TST", name="Test Co")
    db.add(c)
    db.flush()
    return c


def add(db, co, tag, value, start, end, fy, fp, *, unit="USD", accn="a1", form="10-Q",
        filed="2025-02-01", derived=False):
    start = date.fromisoformat(start) if start else None
    end = date.fromisoformat(end)
    row = XbrlFact(
        company_id=co.id, tag=f"us-gaap:{tag}", unit=unit, value=value, period_start=start,
        period_end=end, duration_days=(end - start).days if start else None, fiscal_year=fy,
        fiscal_period=fp, form=form, accession_no=accn, filed=date.fromisoformat(filed),
        derived=derived, source_url=f"https://sec.example/{accn}", fetched_at=datetime.now(UTC),
        sha256="0" * 64, license_note="test")
    db.add(row)
    db.flush()
    return row.id


def test_quarter_ignores_ytd_and_prefers_first_tag(db, co):
    ytd = add(db, co, "RevenueFromContractWithCustomerExcludingAssessedTax", 180,
              "2024-09-29", "2025-03-29", 2025, "Q2")
    q = add(db, co, "RevenueFromContractWithCustomerExcludingAssessedTax", 80,
            "2024-12-29", "2025-03-29", 2025, "Q2")
    add(db, co, "Revenues", 81, "2024-12-29", "2025-03-29", 2025, "Q2")
    f = get_xbrl_fact(db, CIK, "revenue", 2025, "Q2")
    assert (f.value, f.fact_ids, f.unit, f.derived) == (80.0, (q,), "USD", False)
    assert ytd not in f.fact_ids


def test_falls_back_to_later_tag_when_first_has_no_data(db, co):
    fid = add(db, co, "SalesRevenueNet", 50, "2014-12-28", "2015-03-28", 2015, "Q2")
    assert get_xbrl_fact(db, "42", "revenue", 2015, "Q2").fact_ids == (fid,)


def test_fy_returns_twelve_month_value(db, co):
    add(db, co, "NetIncomeLoss", 10, "2025-06-29", "2025-09-27", 2025, "Q4", accn="k", derived=True)
    fy = add(db, co, "NetIncomeLoss", 40, "2024-09-29", "2025-09-27", 2025, "FY", accn="k")
    assert get_xbrl_fact(db, CIK, "net_income", 2025, "FY").fact_ids == (fy,)
    assert get_xbrl_fact(db, CIK, "net_income", 2025, "Q4").derived is True


def test_original_beats_amendment_and_filed_beats_derived(db, co):
    orig = add(db, co, "NetIncomeLoss", 10, "2024-09-29", "2024-12-28", 2025, "Q1")
    add(db, co, "NetIncomeLoss", 11, "2024-09-29", "2024-12-28", 2025, "Q1", accn="a1x",
        form="10-Q/A", filed="2025-06-01")
    assert get_xbrl_fact(db, CIK, "net_income", 2025, "Q1").fact_ids == (orig,)


def test_instant_cash_answers_quarter_and_year_end(db, co):
    q2 = add(db, co, "CashAndCashEquivalentsAtCarryingValue", 7, None, "2025-03-29", 2025, "Q2")
    ye = add(db, co, "CashAndCashEquivalentsAtCarryingValue", 9, None, "2025-09-27", 2025, "FY",
             accn="k")
    assert get_xbrl_fact(db, CIK, "cash", 2025, "Q2").fact_ids == (q2,)
    assert get_xbrl_fact(db, CIK, "cash", 2025, "Q4").fact_ids == (ye,)
    assert get_xbrl_fact(db, CIK, "cash", 2025, "FY").fact_ids == (ye,)


def test_derived_margin_and_fcf_carry_component_ids(db, co):
    r = add(db, co, "RevenueFromContractWithCustomerExcludingAssessedTax", 200,
            "2024-09-29", "2024-12-28", 2025, "Q1")
    g = add(db, co, "GrossProfit", 90, "2024-09-29", "2024-12-28", 2025, "Q1")
    o = add(db, co, "NetCashProvidedByUsedInOperatingActivities", 30, "2024-09-29", "2024-12-28",
            2025, "Q1")
    c = add(db, co, "PaymentsToAcquirePropertyPlantAndEquipment", 4, "2024-09-29", "2024-12-28",
            2025, "Q1")
    gm = get_xbrl_fact(db, CIK, "gross_margin", 2025, "Q1")
    assert (gm.value, gm.unit, gm.fact_ids, gm.derived) == (45.0, "percent", (g, r), False)
    assert [(x.value, x.unit) for x in gm.components] == [(90.0, "USD"), (200.0, "USD")]
    fcf = get_xbrl_fact(db, CIK, "free_cash_flow", 2025, "Q1")
    assert (fcf.value, fcf.unit, fcf.fact_ids) == (26.0, "USD", (o, c))


def test_none_paths(db, co):
    add(db, co, "RevenueFromContractWithCustomerExcludingAssessedTax", 0,
        "2024-09-29", "2024-12-28", 2025, "Q1")
    add(db, co, "GrossProfit", 5, "2024-09-29", "2024-12-28", 2025, "Q1")
    add(db, co, "NetIncomeLoss", 100, "2024-09-29", "2025-03-29", 2025, "Q2")  # YTD only
    assert get_xbrl_fact(db, CIK, "gross_margin", 2025, "Q1") is None  # zero denominator
    assert get_xbrl_fact(db, CIK, "operating_margin", 2025, "Q1") is None  # component missing
    assert get_xbrl_fact(db, CIK, "net_income", 2025, "Q2") is None  # only a 6-month value
    assert get_xbrl_fact(db, CIK, "revenue", 2024, "Q1") is None  # no such period
    assert get_xbrl_fact(db, CIK, "not_a_concept", 2025, "Q1") is None
    assert get_xbrl_fact(db, "0000000099", "revenue", 2025, "Q1") is None  # unknown company


def test_series_oldest_first_quarters_only(db, co):
    tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
    add(db, co, tag, 2, "2024-12-29", "2025-03-29", 2025, "Q2", accn="a2")
    add(db, co, tag, 1, "2024-09-29", "2024-12-28", 2025, "Q1")
    add(db, co, tag, 9, "2024-09-29", "2025-09-27", 2025, "FY", accn="k")
    assert [f.value for f in get_series(db, CIK, "revenue")] == [1.0, 2.0]
    assert [f.value for f in get_series(db, CIK, "revenue", "FY")] == [9.0]

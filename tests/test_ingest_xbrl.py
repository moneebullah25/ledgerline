"""companyfacts parsing: the fy/fp comparative trap, YTD differencing, Q4 derivation."""

from decimal import Decimal

import pytest

from ledgerline.ingest.xbrl import parse_companyfacts

REV = "RevenueFromContractWithCustomerExcludingAssessedTax"
OCF = "NetCashProvidedByUsedInOperatingActivities"


def f(start, end, val, accn, fy, fp, form="10-Q", filed="2025-01-01"):
    d = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        d["start"] = start
    return d


def payload(**tags):
    return {
        "entityName": "X",
        "facts": {"us-gaap": {t: {"units": units} for t, units in tags.items()}},
    }


# fy2025 of a Sept-year-end filer: Q1 10-Q (a1), Q2 10-Q (a2), Q3 10-Q (a3), 10-K (k).
YEAR = {
    REV: {
        "USD": [
            f("2024-09-29", "2024-12-28", 100, "a1", 2025, "Q1"),
            f("2023-10-01", "2023-12-30", 90, "a1", 2025, "Q1"),  # comparative, same fy/fp
            f("2024-12-29", "2025-03-29", 80, "a2", 2025, "Q2"),
            f("2024-09-29", "2025-03-29", 180, "a2", 2025, "Q2"),  # 6-month YTD
            f("2025-03-30", "2025-06-28", 70, "a3", 2025, "Q3"),
            f("2024-09-29", "2025-06-28", 250, "a3", 2025, "Q3"),
            f("2024-09-29", "2025-09-27", 350, "k", 2025, "FY", "10-K"),
            f("2023-10-01", "2024-09-28", 300, "k", 2025, "FY", "10-K"),  # prior-year comparative
        ]
    },
    OCF: {
        "USD": [  # cash flow: YTD only
            f("2024-09-29", "2024-12-28", 30, "a1", 2025, "Q1"),
            f("2024-09-29", "2025-03-29", 50, "a2", 2025, "Q2"),
            f("2024-09-29", "2025-06-28", 65, "a3", 2025, "Q3"),
            f("2024-09-29", "2025-09-27", 100, "k", 2025, "FY", "10-K"),
        ]
    },
    "CashAndCashEquivalentsAtCarryingValue": {
        "USD": [
            f(None, "2025-03-29", 7, "a2", 2025, "Q2"),
            f(None, "2024-09-28", 6, "a2", 2025, "Q2"),  # prior year-end balance in the 10-Q
        ]
    },
    "EarningsPerShareDiluted": {
        "USD/shares": [
            f("2024-09-29", "2025-06-28", 3.1, "a3", 2025, "Q3"),
            f("2024-09-29", "2025-09-27", 4.2, "k", 2025, "FY", "10-K"),
        ]
    },
    "SomeUnmappedTag": {"USD": [f("2024-09-29", "2024-12-28", 1, "a1", 2025, "Q1")]},
}


def rows_by(tag, rows):
    return {
        (r["fiscal_period"], r["duration_days"], r["derived"]): r["value"]
        for r in rows
        if r["tag"] == f"us-gaap:{tag}"
    }


def test_comparatives_dropped_and_labeled_by_own_period():
    rev = rows_by(REV, parse_companyfacts(payload(**YEAR)))
    assert Decimal(90) not in rev.values() and Decimal(300) not in rev.values()
    assert rev[("Q1", 90, False)] == 100
    assert rev[("Q2", 90, False)] == 80 and rev[("Q2", 181, False)] == 180
    assert rev[("FY", 363, False)] == 350


def test_revenue_q4_derived_from_fy_minus_ytd():
    rev = rows_by(REV, parse_companyfacts(payload(**YEAR)))
    assert rev[("Q4", 90, True)] == 100  # 350 - 250
    assert not any(k[2] for k in rev if k[0] != "Q4")  # filed quarters are not re-derived


def test_cash_flow_quarters_from_ytd_differences():
    rows = [r for r in parse_companyfacts(payload(**YEAR)) if r["tag"].endswith(OCF)]
    q = {r["fiscal_period"]: r for r in rows if 80 <= (r["duration_days"] or 0) <= 100}
    assert {p: q[p]["value"] for p in q} == {"Q1": 30, "Q2": 20, "Q3": 15, "Q4": 35}
    assert [q[p]["derived"] for p in ("Q1", "Q2", "Q3", "Q4")] == [False, True, True, True]
    assert str(q["Q3"]["period_start"]) == "2025-03-30" and q["Q3"]["accession_no"] == "a3"


def test_instants_have_no_duration_and_eps_is_not_derived():
    rows = parse_companyfacts(payload(**YEAR))
    cash = [r for r in rows if r["tag"].endswith("CashAndCashEquivalentsAtCarryingValue")]
    assert [(r["value"], r["duration_days"], r["period_start"]) for r in cash] == [(7, None, None)]
    eps = [r for r in rows if r["tag"].endswith("EarningsPerShareDiluted")]
    assert all(not r["derived"] and r["unit"] == "USD/share" for r in eps)
    assert all(r["tag"] != "us-gaap:SomeUnmappedTag" for r in rows)


def test_direct_q4_in_10k_is_labeled_q4_and_not_rederived():
    tags = {
        REV: {
            "USD": YEAR[REV]["USD"] + [f("2025-06-29", "2025-09-27", 101, "k", 2025, "FY", "10-K")]
        }
    }
    rev = rows_by(REV, parse_companyfacts(payload(**tags)))
    assert rev[("Q4", 90, False)] == 101 and ("Q4", 90, True) not in rev


def test_amendment_does_not_replace_original_in_derivation():
    amended = f("2024-09-29", "2025-06-28", 999, "a3x", 2025, "Q3", "10-Q/A", "2025-12-01")
    tags = {OCF: {"USD": YEAR[OCF]["USD"] + [amended]}}
    rows = [r for r in parse_companyfacts(payload(**tags)) if r["derived"]]
    q = {r["fiscal_period"]: r["value"] for r in rows}
    assert q["Q3"] == 15 and q["Q4"] == 35


def test_schema_drift_raises():
    with pytest.raises(KeyError):
        parse_companyfacts({"entityName": "X"})
    with pytest.raises(KeyError):
        parse_companyfacts(payload(SomeUnmappedTag=YEAR["SomeUnmappedTag"]))


def test_derived_quarter_never_duplicates_a_filed_key():
    # PepsiCo-style 12/12/12/16-week year: the 16-week Q4 is tagged under fp=FY and not
    # relabeled (span "other"), and FY - (Q1+Q2+Q3) lands on the same (period, accession)
    # key. Only the filed row stays.
    rows = parse_companyfacts(
        payload(
            **{
                REV: {
                    "USD": [
                        f("2024-09-29", "2024-12-21", 80, "a1", 2025, "Q1"),
                        f("2024-12-22", "2025-03-15", 80, "a2", 2025, "Q2"),
                        f("2025-03-16", "2025-06-07", 90, "a3", 2025, "Q3"),
                        f("2024-09-29", "2025-09-27", 350, "k", 2025, "FY", "10-K"),
                        f("2025-06-08", "2025-09-27", 100, "k", 2025, "FY", "10-K"),
                    ]
                }
            }
        )
    )
    same = [r for r in rows if str(r["period_start"]) == "2025-06-08"]
    assert [(r["value"], r["derived"]) for r in same] == [(100, False)]

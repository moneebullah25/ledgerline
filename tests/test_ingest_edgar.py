"""EDGAR selection, EX-99.1 lookup, release->quarter mapping, and re-run idempotency."""

import json
from datetime import date

import httpx
import pytest
from sqlalchemy import func, select, update

from ledgerline.config import settings
from ledgerline.ingest import edgar, market, xbrl
from ledgerline.ingest.edgar import _release_period
from ledgerline.ingest.http import TransientFetchError, record_failure
from ledgerline.models import Filing, IngestFailure, MarketPrice, XbrlFact

REPORTS = [  # newest first, as EDGAR lists them
    {"accessionNumber": "q3", "form": "10-Q", "reportDate": "2025-06-28"},
    {"accessionNumber": "q2", "form": "10-Q", "reportDate": "2025-03-29"},
    {"accessionNumber": "k", "form": "10-K", "reportDate": "2024-09-28"},
]
PERIODS = {"q3": (2025, "Q3"), "q2": (2025, "Q2"), "k": (2024, "FY")}


def test_release_maps_to_latest_period_end_before_it():
    assert _release_period(date(2025, 7, 31), REPORTS, PERIODS) == (2025, "Q3")
    assert _release_period(date(2025, 5, 1), REPORTS, PERIODS) == (2025, "Q2")
    assert _release_period(date(2024, 10, 31), REPORTS, PERIODS) == (2024, "Q4")  # 10-K -> Q4


def test_release_without_a_recent_period_gets_null():
    # Q4 release before its 10-K exists: nearest period end is 94 days back -> no guess.
    assert _release_period(date(2025, 9, 30), REPORTS, PERIODS) == (None, None)
    assert _release_period(date(2024, 1, 1), REPORTS, PERIODS) == (None, None)


INDEX = b"""<table class="tableFile">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/2/x-8k.htm">x-8k.htm</a></td><td>8-K</td><td>1</td></tr>
<tr><td>2</td><td>EX-99.1</td><td><a href="/Archives/edgar/data/1/2/a2026q2earningsreleaseex-9.htm">a</a></td><td>EX-99.1</td><td>1</td></tr>
</table>"""

SUBMISSIONS = {
    "name": "Apple Inc.",
    "fiscalYearEnd": "0927",
    "filings": {
        "recent": {
            "accessionNumber": ["r1", "q1", "old"],
            "form": ["8-K", "10-Q", "10-Q"],
            "filingDate": ["2025-01-30", "2025-01-31", "2024-08-02"],
            "reportDate": ["2025-01-30", "2024-12-28", "2024-06-29"],
            "primaryDocument": ["r1.htm", "q1.htm", "old.htm"],
            "items": ["2.02,9.01", "", ""],
        }
    },
}
FACTS = {
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "start": "2024-09-29",
                            "end": "2024-12-28",
                            "val": 5,
                            "accn": "q1",
                            "fy": 2025,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2025-01-31",
                        }
                    ]
                }
            }
        }
    },
}


def fake_sec(url: str) -> bytes:
    if "submissions" in url:
        return json.dumps(SUBMISSIONS).encode()
    if "companyfacts" in url:
        return json.dumps(FACTS).encode()
    if url.endswith("-index.html"):
        return INDEX
    if url.endswith("old.htm"):
        from ledgerline.ingest.http import PermanentFetchError

        raise PermanentFetchError(url, 404)
    return b"<p>" + url.encode() + b"</p>"


def test_exhibit_found_by_type_not_name(monkeypatch):
    monkeypatch.setattr(edgar, "sec_get", fake_sec)
    assert (
        edgar._exhibit_99_1("0000000001", "0000000001-26-000002")
        == "a2026q2earningsreleaseex-9.htm"
    )


@pytest.mark.db
def test_second_run_inserts_nothing(db, monkeypatch, tmp_path):
    monkeypatch.setattr(settings(), "data_dir", tmp_path)
    monkeypatch.setattr(edgar, "sec_get", fake_sec)
    monkeypatch.setattr(xbrl, "sec_get", fake_sec)

    assert edgar.ingest_filings(db, "AAPL", quarters=2) == 2  # q1 and the r1 release
    assert xbrl.ingest_xbrl_facts(db, "AAPL") == 1
    rows = {f.accession_no: f for f in db.scalars(select(Filing))}
    assert rows["r1"].doc_url.endswith("a2026q2earningsreleaseex-9.htm")
    assert (rows["r1"].fiscal_year, rows["r1"].fiscal_period) == (2025, "Q1")
    assert (rows["q1"].fiscal_year, rows["q1"].fiscal_period) == (2025, "Q1")
    # The 404 on "old" became one failure row, not a crash.
    fails = db.scalars(select(IngestFailure)).all()
    assert [(f.key, f.error_class) for f in fails] == [("AAPL:old", "http_4xx")]

    assert edgar.ingest_filings(db, "AAPL", quarters=2) == 0
    assert xbrl.ingest_xbrl_facts(db, "AAPL") == 0
    assert db.scalar(select(func.count()).select_from(Filing)) == 2
    assert db.scalar(select(func.count()).select_from(XbrlFact)) == 1
    assert db.scalar(select(func.count()).select_from(IngestFailure)) == 1  # not duplicated


def test_untagged_newest_filing_steps_the_fiscal_cycle():
    periods = {"q2": (2025, "Q2"), "k": (2024, "FY")}  # q3 not in companyfacts yet
    edgar._fill_untagged(REPORTS, periods)
    assert periods["q3"] == (2025, "Q3")
    periods = {"k": (2024, "FY")}  # k -> q2 skips a quarter: no guess, and none after it
    edgar._fill_untagged(REPORTS, periods)
    assert periods == {"k": (2024, "FY")}


@pytest.mark.db
def test_store_is_idempotent_at_the_database(db, monkeypatch, tmp_path):
    # Bypasses ingest_filings' `known` pre-filter: the unique accession_no alone must hold.
    monkeypatch.setattr(settings(), "data_dir", tmp_path)
    monkeypatch.setattr(edgar, "sec_get", fake_sec)
    company_id = edgar.ensure_company(db, "AAPL", "Apple Inc.")
    r = {
        "accessionNumber": "q1",
        "form": "10-Q",
        "filingDate": "2025-01-31",
        "reportDate": "2024-12-28",
    }
    assert edgar._store(db, company_id, "0000320193", r, "q1.htm", 2025, "Q1") == 1
    assert edgar._store(db, company_id, "0000320193", r, "q1.htm", 2025, "Q1") == 0
    assert db.scalar(select(func.count()).select_from(Filing)) == 1


@pytest.mark.db
def test_record_failure_is_one_row_without_a_pre_check(db):
    for _ in range(2):
        record_failure(db, "ingest_edgar_filings", "AAPL:x", ValueError("bad"))
    record_failure(db, "ingest_edgar_filings", "AAPL:x", KeyError("k"))  # another class
    rows = db.execute(select(IngestFailure.key, IngestFailure.error_class)).all()
    assert sorted(rows) == [("AAPL:x", "parse"), ("AAPL:x", "schema_drift")]


@pytest.mark.db
@pytest.mark.parametrize("exc", [TransientFetchError("HTTP 503"), httpx.ConnectError("reset")])
def test_exhausted_retries_become_a_row_and_the_ticker_continues(db, monkeypatch, tmp_path, exc):
    monkeypatch.setattr(settings(), "data_dir", tmp_path)

    def flaky(url):
        if url.endswith("q1.htm"):
            raise exc
        return fake_sec(url)

    monkeypatch.setattr(edgar, "sec_get", flaky)
    assert edgar.ingest_filings(db, "AAPL", quarters=2) == 1  # r1 still stored
    fails = db.execute(select(IngestFailure.key, IngestFailure.error_class)).all()
    assert ("AAPL:q1", "http_transient_exhausted") in fails


def ocf(*facts):
    return {
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": list(facts)}}
            }
        },
    }


Q1 = {
    "start": "2024-09-29",
    "end": "2024-12-28",
    "val": 30,
    "accn": "a1",
    "fy": 2025,
    "fp": "Q1",
    "form": "10-Q",
    "filed": "2025-01-31",
}
H1 = {
    "start": "2024-09-29",
    "end": "2025-03-29",
    "val": 50,
    "accn": "a2",
    "fy": 2025,
    "fp": "Q2",
    "form": "10-Q",
    "filed": "2025-05-01",
}


@pytest.mark.db
def test_changed_payload_rebases_derived_quarters(db, monkeypatch, tmp_path):
    monkeypatch.setattr(settings(), "data_dir", tmp_path)
    payload = {}
    monkeypatch.setattr(xbrl, "sec_get", lambda url: json.dumps(payload).encode())

    def derived():
        return db.execute(
            select(XbrlFact.fiscal_period, XbrlFact.value).where(XbrlFact.derived)
        ).all()

    payload = ocf(Q1, H1)
    assert xbrl.ingest_xbrl_facts(db, "AAPL") == 3  # Q1, H1, derived Q2 = 20
    assert derived() == [("Q2", 20)]
    assert xbrl.ingest_xbrl_facts(db, "AAPL") == 0

    payload = ocf(Q1, H1 | {"val": 55})  # the YTD base moved (restated in place)
    assert xbrl.ingest_xbrl_facts(db, "AAPL") == 2  # H1 and the derived Q2 updated
    assert derived() == [("Q2", 25)]

    # A stored row a parser fix would label differently (same value) is corrected too.
    db.execute(update(XbrlFact).where(XbrlFact.derived).values(duration_days=1))
    assert xbrl.ingest_xbrl_facts(db, "AAPL") == 1
    assert derived() == [("Q2", 25)]

    # An amendment files the 3-month Q2 itself: the derived stand-in is deleted, not kept.
    q2 = H1 | {"start": "2024-12-29", "val": 24, "accn": "a2x", "form": "10-Q/A"}
    payload = ocf(Q1, H1 | {"val": 55}, q2)
    assert xbrl.ingest_xbrl_facts(db, "AAPL") == 2  # q2 inserted, derived Q2 deleted
    assert derived() == []
    assert db.scalar(select(func.count()).select_from(XbrlFact)) == 3


def chart(adj):
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {"gmtoffset": -14400},
                        "timestamp": [1756992600],
                        "indicators": {
                            "quote": [{"close": [239.78], "volume": [100]}],
                            "adjclose": [{"adjclose": [adj]}],
                        },
                    }
                ]
            }
        }
    ).encode()


@pytest.mark.db
def test_market_prices_rebased_on_refetch(db, monkeypatch):
    body = chart(238.9)
    monkeypatch.setattr(market, "get", lambda url, headers: body)
    day = date(2025, 9, 4)
    assert market.ingest_market_data(db, "AAPL", day, day) == 1
    assert market.ingest_market_data(db, "AAPL", day, day) == 0
    body = chart(238.90001)  # Yahoo's float32 ulp wobble between calls is not a re-base
    assert market.ingest_market_data(db, "AAPL", day, day) == 0
    body = chart(237.5)  # a later dividend re-based the history
    assert market.ingest_market_data(db, "AAPL", day, day) == 1
    row = db.scalars(select(MarketPrice)).one()
    assert row.adj_close == 237.5 and row.sha256 != "unrecorded"
    db.execute(update(MarketPrice).values(sha256="unrecorded", volume=None))  # pre-0004 row
    assert market.ingest_market_data(db, "AAPL", day, day) == 1
    db.refresh(row)
    assert row.sha256 != "unrecorded" and row.volume == 100

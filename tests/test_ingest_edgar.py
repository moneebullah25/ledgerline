"""EDGAR selection, EX-99.1 lookup, release->quarter mapping, and re-run idempotency."""

import json
from datetime import date

import pytest
from sqlalchemy import func, select

from ledgerline.config import settings
from ledgerline.ingest import edgar, xbrl
from ledgerline.ingest.edgar import _release_period
from ledgerline.models import Filing, IngestFailure, XbrlFact

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

SUBMISSIONS = {"name": "Apple Inc.", "fiscalYearEnd": "0927", "filings": {"recent": {
    "accessionNumber": ["r1", "q1", "old"], "form": ["8-K", "10-Q", "10-Q"],
    "filingDate": ["2025-01-30", "2025-01-31", "2024-08-02"],
    "reportDate": ["2025-01-30", "2024-12-28", "2024-06-29"],
    "primaryDocument": ["r1.htm", "q1.htm", "old.htm"], "items": ["2.02,9.01", "", ""]}}}
FACTS = {"entityName": "Apple Inc.", "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
    {"start": "2024-09-29", "end": "2024-12-28", "val": 5, "accn": "q1", "fy": 2025, "fp": "Q1",
     "form": "10-Q", "filed": "2025-01-31"}]}}}}}


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
    assert edgar._exhibit_99_1("0000000001", "0000000001-26-000002") == \
        "a2026q2earningsreleaseex-9.htm"


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

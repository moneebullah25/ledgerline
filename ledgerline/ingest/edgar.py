"""EDGAR filings (FR-1): 10-Q/10-K primary documents and 8-K Item 2.02 earnings releases.

For an earnings 8-K the stored document is exhibit 99.1 (the press release), since that
is where management's claims are; the 8-K cover page carries nothing to verify.
"""

import json
import re
from datetime import UTC, date, datetime
from pathlib import PurePosixPath

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ledgerline.concepts import UNIVERSE
from ledgerline.ingest.http import (
    PermanentFetchError,
    TransientFetchError,
    record_failure,
    sec_get,
)
from ledgerline.ingest.objects import put_object
from ledgerline.models import Company, Filing

LICENSE = "SEC EDGAR public filing"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EXTRA_RELEASES = 2  # releases reach back further than 10-Q/10-Ks: guidance needs a prior quarter
MAX_RELEASE_LAG_DAYS = 75  # a release comes out within ~2-8 weeks of the quarter it reports


def ensure_company(s: Session, ticker: str, name: str, fiscal_year_end: str | None = None) -> int:
    stmt = insert(Company).values(
        cik=UNIVERSE[ticker], ticker=ticker, name=name, fiscal_year_end=fiscal_year_end
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Company.cik],
        set_={"fiscal_year_end": stmt.excluded.fiscal_year_end}
        if fiscal_year_end
        else {"name": Company.name},  # no-op update so RETURNING yields the id
    ).returning(Company.id)
    return s.scalar(stmt)


def ingest_filings(s: Session, ticker: str, quarters: int = 6) -> int:
    cik = UNIVERSE[ticker]
    sub = json.loads(sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    company_id = ensure_company(s, ticker, sub["name"], sub.get("fiscalYearEnd"))
    rows = _recent(sub["filings"]["recent"])
    periods = _filing_periods(cik)
    known = set(s.scalars(select(Filing.accession_no).where(Filing.company_id == company_id)))

    reports = [r for r in rows if r["form"] in ("10-Q", "10-K")]
    _fill_untagged(reports, periods)
    releases = [r for r in rows if r["form"] == "8-K" and "2.02" in r["items"].split(",")]
    todo = reports[:quarters] + releases[: quarters + EXTRA_RELEASES]
    inserted = 0
    for r in todo:
        if r["accessionNumber"] in known:
            continue  # idempotent without refetching the document
        try:
            if r["form"] == "8-K":
                doc = _exhibit_99_1(cik, r["accessionNumber"])
                fy, fp = _release_period(date.fromisoformat(r["filingDate"]), reports, periods)
            else:
                doc = r["primaryDocument"]
                fy, fp = periods.get(r["accessionNumber"], (None, None))
            inserted += _store(s, company_id, cik, r, doc, fy, fp)
        # Retries exhausted on one document: record it and move on to the next filing.
        except (
            PermanentFetchError,
            TransientFetchError,
            httpx.TransportError,
            LookupError,
            ValueError,
        ) as e:
            record_failure(s, "ingest_edgar_filings", f"{ticker}:{r['accessionNumber']}", e)
    return inserted


def _recent(recent: dict) -> list[dict]:
    """Column arrays -> row dicts, newest first (EDGAR's order)."""
    keys = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument", "items")
    return [dict(zip(keys, vals)) for vals in zip(*(recent[k] for k in keys))]


def _filing_periods(cik: str) -> dict[str, tuple[int, str]]:
    """accession -> (fy, fp) as the filer tagged it (dei:DocumentFiscalYearFocus/PeriodFocus).

    Taken from companyfacts rather than computed from the calendar because fiscal-year
    naming is company-specific: Home Depot calls the year ending Feb 2025 "fiscal 2024".
    """
    facts = json.loads(sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"))
    out: dict[str, tuple[int, str]] = {}
    for tag in facts["facts"].get("us-gaap", {}).values():
        for rows in tag["units"].values():
            for f in rows:
                if f.get("fy") and f.get("fp"):
                    out.setdefault(f["accn"], (f["fy"], f["fp"]))
    return out


NEXT_FP = {"Q1": "Q2", "Q2": "Q3", "Q3": "FY", "FY": "Q1"}


def _fill_untagged(reports: list[dict], periods: dict[str, tuple[int, str]]) -> None:
    """companyfacts lags EDGAR by days to weeks, so the newest 10-Q may have no fy/fp yet.
    Such a filing gets the fiscal period after the previous filing's: 10-Q/10-K filings
    come one per quarter, in order. Without a tagged predecessor exactly one quarter earlier
    it stays unlabeled rather than guessed."""
    for prev, r in zip(reports[::-1], reports[::-1][1:]):  # oldest first
        gap = (
            date.fromisoformat(r["reportDate"] or "1900-01-01")
            - date.fromisoformat(prev["reportDate"] or "2999-01-01")
        ).days
        if (
            r["accessionNumber"] not in periods
            and prev["accessionNumber"] in periods
            and 80 <= gap <= 100
        ):
            fy, fp = periods[prev["accessionNumber"]]
            periods[r["accessionNumber"]] = (fy + 1 if fp == "FY" else fy, NEXT_FP[fp])


def _release_period(
    filed: date, reports: list[dict], periods: dict[str, tuple[int, str]]
) -> tuple[int | None, str | None]:
    """The quarter an earnings release reports: the latest 10-Q/10-K period end on or before
    the release date. The release usually precedes that 10-Q by days, but the period end
    always precedes the release. A release more than MAX_RELEASE_LAG_DAYS after the nearest
    known period end (its 10-Q not filed yet, or an off-cycle 2.02) gets NULL, never a guess.
    A release matched to a 10-K is labeled Q4: it reports the fourth quarter and the year.
    """
    best = max(
        (r for r in reports if r["reportDate"] and date.fromisoformat(r["reportDate"]) <= filed),
        key=lambda r: r["reportDate"],
        default=None,
    )
    if best is None or (filed - date.fromisoformat(best["reportDate"])).days > MAX_RELEASE_LAG_DAYS:
        return None, None
    fy, fp = periods.get(best["accessionNumber"], (None, None))
    return fy, ("Q4" if fp == "FY" else fp)


def _exhibit_99_1(cik: str, accession: str) -> str:
    # index.json lists file names but not document types, and names are unreliable
    # (Coca-Cola's EX-99.1 is "a2026q2earningsreleaseex-9.htm"), so read the Type column.
    url = f"{ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/{accession}-index.html"
    soup = BeautifulSoup(sec_get(url), "lxml")
    docs: list[tuple[str, str]] = []
    for tr in soup.select("table.tableFile tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        link = tr.find("a")
        if len(cells) >= 4 and link:
            docs.append((cells[3].upper(), PurePosixPath(link["href"]).name))
    for want in (re.compile(r"EX-99\.0*1$"), re.compile(r"EX-99")):
        for typ, name in docs:
            if want.match(typ) and name.lower().endswith((".htm", ".html", ".txt")):
                return name
    raise LookupError(f"no EX-99.1 in {url}")


def _store(
    s: Session, company_id: int, cik: str, r: dict, doc: str, fy: int | None, fp: str | None
) -> int:
    url = f"{ARCHIVES}/{int(cik)}/{r['accessionNumber'].replace('-', '')}/{doc}"
    data = sec_get(url)
    sha, path = put_object(data, PurePosixPath(doc).suffix.lower())
    stmt = (
        insert(Filing)
        .values(
            company_id=company_id,
            accession_no=r["accessionNumber"],
            form_type=r["form"],
            filed_at=date.fromisoformat(r["filingDate"]),
            period_of_report=date.fromisoformat(r["reportDate"]) if r["reportDate"] else None,
            fiscal_year=fy,
            fiscal_period=fp,
            doc_url=url,
            storage_path=path,
            source_url=url,
            fetched_at=datetime.now(UTC),
            sha256=sha,
            license_note=LICENSE,
        )
        .on_conflict_do_nothing(index_elements=[Filing.accession_no])
        .returning(Filing.id)
    )
    return 1 if s.scalar(stmt) is not None else 0

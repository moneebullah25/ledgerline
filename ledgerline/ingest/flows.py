"""The five §10.1 Prefect flows: thin schedulable wrappers over the plain ingest functions.

Each company runs in its own transaction; a failure rolls back that company only and is
stored as an ingest_failure row, so one bad filer never stops the rest (NFR-4).
"""

import logging
from collections.abc import Callable
from datetime import date, timedelta

from prefect import flow
from sqlalchemy import select

from ledgerline.concepts import UNIVERSE
from ledgerline.db import session
from ledgerline.ingest import edgar, market, xbrl
from ledgerline.ingest.http import record_failure
from ledgerline.models import Company

log = logging.getLogger(__name__)
PRICE_YEARS = 3


def guarded(flow_name: str, key: str, fn: Callable[..., int], *args) -> int | None:
    try:
        with session() as s:
            return fn(s, *args)
    except Exception as e:  # noqa: BLE001 - recorded as data, by design
        log.warning("%s %s failed: %r", flow_name, key, e)
        with session() as s:
            record_failure(s, flow_name, key, e)
        return None


def run_filings(tickers: list[str], quarters: int = 6) -> dict[str, int | None]:
    return {t: guarded("ingest_edgar_filings", t, edgar.ingest_filings, t, quarters) for t in tickers}


def run_facts(tickers: list[str]) -> dict[str, int | None]:
    return {t: guarded("ingest_xbrl_facts", t, xbrl.ingest_xbrl_facts, t) for t in tickers}


def run_prices(tickers: list[str], start: date | None = None,
               end: date | None = None) -> dict[str, int | None]:
    end = end or date.today()
    start = start or end - timedelta(days=365 * PRICE_YEARS)
    return {t: guarded("ingest_market_data", t, market.ingest_market_data, t, start, end)
            for t in dict.fromkeys([*tickers, market.BENCHMARK])}


@flow(name="ingest_edgar_filings")
def ingest_edgar_filings(tickers: list[str] | None = None, quarters: int = 6) -> dict[str, int | None]:
    return run_filings(tickers or list(UNIVERSE), quarters)


@flow(name="ingest_xbrl_facts")
def ingest_xbrl_facts(tickers: list[str] | None = None) -> dict[str, int | None]:
    return run_facts(tickers or list(UNIVERSE))


@flow(name="ingest_market_data")
def ingest_market_data(tickers: list[str] | None = None, start: date | None = None,
                       end: date | None = None) -> dict[str, int | None]:
    return run_prices(tickers or list(UNIVERSE), start, end)


@flow(name="ingest_call_audio")
def ingest_call_audio(path_or_url: str, kind: str, ticker: str | None = None,
                      event_date: date | None = None, license_note: str = "",
                      fiscal_year: int | None = None,
                      fiscal_period: str | None = None) -> int | None:
    """fiscal_year/fiscal_period tie an earnings call to its report quarter (the pipeline
    selects calls by them)."""
    from ledgerline.ingest import call_audio  # owned by S2; imported late on purpose

    def run(s):
        company_id = s.scalar(select(Company.id).where(Company.ticker == ticker)) if ticker else None
        return call_audio.ingest_call_audio(s, path_or_url, kind=kind, company_id=company_id,
                                            event_date=event_date, license_note=license_note,
                                            fiscal_year=fiscal_year, fiscal_period=fiscal_period)
    return guarded("ingest_call_audio", path_or_url, run)


@flow(name="ingest_fomc_eval_set")
def ingest_fomc_eval_set(dates: list[str]) -> int | None:
    from ledgerline.ingest import fomc  # owned by S2; imported late on purpose

    return guarded("ingest_fomc_eval_set", ",".join(dates),
                   lambda s: len(fomc.ingest_fomc_eval_set(s, dates)))

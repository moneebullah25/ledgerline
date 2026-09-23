"""Daily closes from Yahoo's chart API, for the market-reaction agent (SPY is the benchmark)."""

import json
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ledgerline.ingest.http import get
from ledgerline.models import MarketPrice

BENCHMARK = "SPY"
# Not an SEC call. Yahoo answers 429 to library User-Agents and, oddly, to full browser
# strings without its cookies; the bare "Mozilla/5.0" is served.
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def ingest_market_data(s: Session, ticker: str, start: date, end: date) -> int:
    p1 = int(datetime.combine(start, time(), UTC).timestamp())
    p2 = int(datetime.combine(end + timedelta(days=1), time(), UTC).timestamp())
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={p2}&interval=1d")
    rows = parse_chart(json.loads(get(url, _HEADERS)), ticker, url)
    if not rows:
        return 0
    # ponytail: first write wins, so adj_close is not re-based after later dividends/splits;
    # delete and reload a ticker if abnormal returns ever span a corporate action.
    stmt = insert(MarketPrice).on_conflict_do_nothing(constraint="uq_price").returning(MarketPrice.id)
    return len(s.execute(stmt, rows).all())


def parse_chart(payload: dict, ticker: str, url: str) -> list[dict]:
    r = payload["chart"]["result"][0]
    offset = r["meta"]["gmtoffset"]  # timestamps are the session open; shift to exchange-local
    quote, adj = r["indicators"]["quote"][0], r["indicators"]["adjclose"][0]["adjclose"]
    now = datetime.now(UTC)
    rows = {}
    for i, ts in enumerate(r.get("timestamp", [])):
        if quote["close"][i] is None or adj[i] is None:
            continue  # halted or not-yet-settled bar
        day = datetime.fromtimestamp(ts + offset, UTC).date()
        rows[day] = {"ticker": ticker, "day": day, "close": quote["close"][i],
                     "adj_close": adj[i], "volume": quote["volume"][i],
                     "source_url": url, "fetched_at": now}
    return list(rows.values())

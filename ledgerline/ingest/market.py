"""Daily closes from Yahoo's chart API, for the market-reaction agent (SPY is the benchmark)."""

import hashlib
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
LICENSE_NOTE = ("Yahoo Finance chart API; personal/research use under Yahoo's terms, "
                "not redistributed")


def ingest_market_data(s: Session, ticker: str, start: date, end: date) -> int:
    p1 = int(datetime.combine(start, time(), UTC).timestamp())
    p2 = int(datetime.combine(end + timedelta(days=1), time(), UTC).timestamp())
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={p2}&interval=1d")
    raw = get(url, _HEADERS)
    prov = {"sha256": hashlib.sha256(raw).hexdigest(), "license_note": LICENSE_NOTE}  # FR-6
    rows = [r | prov for r in parse_chart(json.loads(raw), ticker, url)]
    if not rows:
        return 0
    # Yahoo re-bases the whole history after a split or dividend (close for splits, adj_close
    # for both), so every re-fetched day is upserted: stored returns follow the latest basis.
    # Returns rows inserted or re-based.
    stmt = insert(MarketPrice)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_price",
        set_={c: stmt.excluded[c] for c in ("close", "adj_close", "volume", "source_url",
                                            "fetched_at", "sha256", "license_note")},
        where=(MarketPrice.close != stmt.excluded.close)
        | (MarketPrice.adj_close != stmt.excluded.adj_close),
    ).returning(MarketPrice.id)
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

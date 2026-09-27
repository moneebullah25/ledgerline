"""MarketReaction agent: context for severity, never a truth signal.

Abnormal return = stock return minus SPY return over the [-1, +1] trading-day window around
the event (day 0 = first trading day on or after the release/call date): adj_close at day
+1 over adj_close at day -2, which covers an after-close release reacting on day +1.
Stance is always `insufficient`: a price move cannot verify what management said.
"""

from datetime import date, timedelta
from threading import Lock

from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline.graph.state import Ctx
from ledgerline.llm import OFFLINE, Usage
from ledgerline.models import Claim, MarketPrice
from ledgerline.schemas import Citation, EvidenceItem

BENCH = "SPY"
_PAD = timedelta(days=14)  # calendar days fetched either side; covers holidays


def run(s: Session, claim: Claim, ctx: Ctx, lock: Lock) -> tuple[EvidenceItem, Usage]:
    with lock:
        rows = s.scalars(
            select(MarketPrice).where(
                MarketPrice.ticker.in_((ctx.ticker, BENCH)),
                MarketPrice.day.between(ctx.event_date - _PAD, ctx.event_date + _PAD),
            )
        ).all()
        prices = [(r.id, r.ticker, r.day, r.adj_close) for r in rows]
    return assess(ctx.ticker, ctx.event_date, prices), OFFLINE


def assess(ticker: str, event: date, prices: list[tuple[int, str, date, float]]) -> EvidenceItem:
    """prices: (row id, ticker, day, adj_close). Only days both series traded are used."""
    by = {t: {d: (i, p) for i, tk, d, p in prices if tk == t} for t in (ticker, BENCH)}
    days = sorted(by[ticker].keys() & by[BENCH].keys())
    i0 = next((i for i, d in enumerate(days) if d >= event), None)
    if i0 is None or i0 < 2 or i0 + 1 >= len(days):
        return EvidenceItem(
            agent="market_reaction",
            stance="insufficient",
            summary="not enough trading days around the event for a [-1, +1] window",
        )
    start, end = days[i0 - 2], days[i0 + 1]
    rets, cites = {}, []
    for t in (ticker, BENCH):
        (a_id, a), (b_id, b) = by[t][start], by[t][end]
        rets[t] = b / a - 1
        cites += [
            Citation(kind="price", ref_id=a_id, value=a, unit="USD"),
            Citation(kind="price", ref_id=b_id, value=b, unit="USD"),
        ]
    ar = rets[ticker] - rets[BENCH]
    return EvidenceItem(
        agent="market_reaction",
        stance="insufficient",
        summary=f"{ticker} abnormal return vs {BENCH} over the [-1, +1] window around {event}.",
        citations=cites,
        payload={
            "abnormal_return": ar,
            "stock_return": rets[ticker],
            "bench_return": rets[BENCH],
            "window": [start.isoformat(), end.isoformat()],
        },
    )

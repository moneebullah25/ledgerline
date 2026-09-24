"""XBRL facts from SEC companyfacts (FR-2), restricted to tags in concepts.TAG_TO_CONCEPT.

Three traps shape this module:

1. companyfacts `fy`/`fp` describe the FILING, not the value. A fy2026 Q3 10-Q also
   carries the fy2025 Q3 comparative under fy=2026 fp=Q3. Rule used here: a filing's own
   period is the latest period end among all of its mapped facts; only facts ending on
   that date keep the filing's fy/fp. Comparatives are dropped, since each period is also
   reported as the current period of its own filing. (Accession-wide rather than per-tag,
   so a tag a filing shows only as a comparative is not mistaken for its current value.)
2. 10-Qs report cash-flow items year-to-date only, and nobody files a Q4 10-Q. A missing
   3-month value is derived as YTD(q) - YTD(q-1), and Q4 as FY - YTD(Q3), with
   derived=True. EPS is not additive, so it is never derived.
3. A fp=FY filing's 3-month value (some 10-Ks tag Q4 directly) is labeled Q4 by duration.
"""

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ledgerline.concepts import TAG_TO_CONCEPT, UNIVERSE
from ledgerline.ingest.edgar import ARCHIVES, ensure_company
from ledgerline.ingest.http import record_failure, sec_get
from ledgerline.ingest.objects import put_object
from ledgerline.models import XbrlFact

LICENSE = "SEC EDGAR XBRL companyfacts (public filing data)"
FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")
UNITS = {"USD": "USD", "USD/shares": "USD/share"}
NOT_ADDITIVE = {"us-gaap:EarningsPerShareDiluted"}
# Duration classes in days (end - start). 52/53-week years put quarters at 83-98 days.
SPANS = {"Q": (80, 100), "H": (170, 190), "9M": (260, 285), "Y": (350, 380)}
YTD_SPAN = {"Q1": "Q", "Q2": "H", "Q3": "9M", "Q4": "Y"}  # Q4's cumulative value is FY
PREV = {"Q2": "Q1", "Q3": "Q2", "Q4": "Q3"}
_CONTENT = ("value", "derived", "fiscal_year", "fiscal_period", "duration_days", "form", "filed")


def ingest_xbrl_facts(s: Session, ticker: str) -> int:
    """Rows inserted, changed or deleted; 0 when re-run over an unchanged payload."""
    cik = UNIVERSE[ticker]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    raw = sec_get(url)
    payload = json.loads(raw)
    sha, _ = put_object(raw, ".json")
    company_id = ensure_company(s, ticker, payload["entityName"])
    rows = parse_companyfacts(payload)

    # A tag we loaded before that is gone from the payload is schema drift, not silence.
    present = {r["tag"] for r in rows}
    stored = set(s.scalars(select(XbrlFact.tag).where(XbrlFact.company_id == company_id).distinct()))
    for tag in sorted(stored - present):
        record_failure(s, "ingest_xbrl_facts", f"{ticker}:{tag}", f"{tag} vanished from {url}",
                       cls="schema_drift")

    now = datetime.now(UTC)
    for r in rows:
        # source_url is the filing a reader can open to check the number; sha256 is of the
        # companyfacts payload the value was actually read from (kept in the object store).
        acc = r["accession_no"]
        r |= {"company_id": company_id, "fetched_at": now, "sha256": sha, "license_note": LICENSE,
              "source_url": f"{ARCHIVES}/{int(cik)}/{acc.replace('-', '')}/{acc}-index.html"}
    # A changed payload (a corrected value, an amendment that moves a YTD base) or a parser
    # fix must not leave a stale row behind: rows whose parsed content differs in any column
    # are updated, and derived quarters the new payload no longer produces are deleted.
    # sha256/fetched_at are not compared (the payload hash changes daily), only refreshed.
    stmt = insert(XbrlFact)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_xbrl_fact",
        set_={c: stmt.excluded[c] for c in _CONTENT + ("fetched_at", "sha256")},
        where=tuple_(*(getattr(XbrlFact, c) for c in _CONTENT))
        .is_distinct_from(tuple_(*(stmt.excluded[c] for c in _CONTENT))),
    ).returning(XbrlFact.id)
    written = sum(len(s.execute(stmt, rows[i:i + 2000]).all()) for i in range(0, len(rows), 2000))

    derived = {_key(r) for r in rows if r["derived"]}
    existing = s.execute(select(XbrlFact.id, XbrlFact.tag, XbrlFact.unit, XbrlFact.period_start,
                                XbrlFact.period_end, XbrlFact.accession_no)
                         .where(XbrlFact.company_id == company_id, XbrlFact.derived))
    stale = [i for i, *k in existing if tuple(k) not in derived]
    if stale:
        s.execute(delete(XbrlFact).where(XbrlFact.id.in_(stale)))
    return written + len(stale)


def span(start: date | None, end: date) -> str | None:
    if start is None:
        return None
    days = (end - start).days
    return next((k for k, (lo, hi) in SPANS.items() if lo <= days <= hi), "other")


def parse_companyfacts(payload: dict) -> list[dict]:
    """companyfacts JSON -> xbrl_fact row dicts (current-period facts plus derived quarters)."""
    gaap = payload["facts"].get("us-gaap", {})
    facts = []
    for tag in TAG_TO_CONCEPT:
        node = gaap.get(tag.split(":", 1)[1])
        if node is None:
            continue
        for unit, items in node["units"].items():
            if unit not in UNITS:
                continue
            for f in items:
                if f["form"] not in FORMS or not f.get("fy") or not f.get("fp"):
                    continue
                start = date.fromisoformat(f["start"]) if "start" in f else None
                end = date.fromisoformat(f["end"])
                facts.append({
                    "tag": tag, "unit": UNITS[unit], "value": Decimal(str(f["val"])),
                    "period_start": start, "period_end": end,
                    "duration_days": (end - start).days if start else None,
                    "fiscal_year": f["fy"], "fiscal_period": f["fp"], "form": f["form"],
                    "accession_no": f["accn"], "filed": date.fromisoformat(f["filed"]),
                    "derived": False,
                })
    if not facts:
        raise KeyError("no mapped us-gaap tags in companyfacts payload")

    current_end: dict[str, date] = {}
    for f in facts:
        current_end[f["accession_no"]] = max(current_end.get(f["accession_no"], f["period_end"]),
                                             f["period_end"])
    rows: dict[tuple, dict] = {}
    for f in facts:
        if f["period_end"] != current_end[f["accession_no"]]:
            continue  # prior-period comparative: fy/fp belong to the filing, not to it
        if f["fiscal_period"] == "FY" and span(f["period_start"], f["period_end"]) == "Q":
            f["fiscal_period"] = "Q4"
        rows.setdefault(_key(f), f)
    out = list(rows.values())
    # A derived quarter can coincide with a filed row (PepsiCo tags its 16-week Q4 under
    # fp=FY): the filed row is the one kept, as the unique key allows only one.
    return out + [d for d in _derive_quarters(out) if _key(d) not in rows]


def _key(f: dict) -> tuple:
    """uq_xbrl_fact within one company."""
    return (f["tag"], f["unit"], f["period_start"], f["period_end"], f["accession_no"])


def _original_first(f: dict) -> tuple:
    # Same value filed more than once (original and amendment): the original, as first
    # filed, is what the market saw on the release date.
    return (f["form"].endswith("/A"), f["filed"], f["accession_no"])


def _derive_quarters(facts: list[dict]) -> list[dict]:
    """Fill missing 3-month values from year-to-date differences (see module docstring)."""
    # (tag, unit, fy) -> {("Q", "Q2"): fact, ("H", "Q2"): fact, ...}. Iterating worst-first
    # means the last write, which wins, is the earliest original filing.
    by_year: dict[tuple, dict[tuple, dict]] = {}
    for f in sorted(facts, key=_original_first, reverse=True):
        kind = span(f["period_start"], f["period_end"])
        if kind is None or f["tag"] in NOT_ADDITIVE:
            continue
        fp = "Q4" if f["fiscal_period"] == "FY" else f["fiscal_period"]
        by_year.setdefault((f["tag"], f["unit"], f["fiscal_year"]), {})[(kind, fp)] = f

    derived = []
    for got in by_year.values():
        quarter = {q: got.get(("Q", q)) for q in YTD_SPAN}
        cum = {"Q1": quarter["Q1"]}
        for q, prev in PREV.items():
            ytd = got.get((YTD_SPAN[q], q))
            before = cum.get(prev)
            if quarter[q] is None and ytd and before and ytd["period_start"] == before["period_start"]:
                quarter[q] = {
                    **ytd, "value": ytd["value"] - before["value"], "derived": True,
                    "period_start": before["period_end"] + timedelta(days=1),
                    "fiscal_period": q,
                }
                quarter[q]["duration_days"] = (ytd["period_end"] - quarter[q]["period_start"]).days
                derived.append(quarter[q])
            # Cumulative through q: as filed, else built from the previous cumulative.
            if ytd:
                cum[q] = ytd
            elif before and quarter[q]:
                cum[q] = {**quarter[q], "value": before["value"] + quarter[q]["value"],
                          "period_start": before["period_start"]}
    return derived

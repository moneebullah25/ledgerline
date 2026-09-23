"""XBRLTable agent (FR-22 typed query). Compares a claim with filed XBRL facts only; it
never reads a number out of text.

Tolerance, level claims: |actual - claimed| <= REL_TOL * |actual| + half a unit in the last
digit the claim printed. "$94.9 billion" printed one decimal of billions, so the rounding
part is 0.05e9: it is consistent with anything in [94.85e9, 94.95e9] plus 0.5% slack for
the gap between release rounding and the filed figure.

Tolerance, change claims (yoy/qoq): |actual_pct - claimed_pct| <= PP_TOL + half a unit in
the last printed digit of the percent ("grew 5%" allows 4.4..5.6 with PP_TOL=0.1).
For margin concepts (already percent) the change is the difference in points.

Guidance whose target period has been reported is checked the same way: supports means
the reported value landed inside the guided value/range.

payload["deviation"] feeds severity: for levels, (nearest claimed value - actual) / |actual|;
for changes, (claimed - actual) in points / 100. payload["tol_used"] = |error| / tolerance.
"""

import math
import re
from threading import Lock

from sqlalchemy.orm import Session

from ledgerline.concepts import CONCEPTS
from ledgerline.graph.state import Ctx
from ledgerline.llm import OFFLINE, Usage
from ledgerline.models import Claim
from ledgerline.schemas import Citation, EvidenceItem
from ledgerline.store.xbrl_query import Fact, get_xbrl_fact

REL_TOL = 0.005
PP_TOL = 0.1
_SCALES = (1.0, 1e3, 1e6, 1e9, 1e12, 0.01)
_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.(\d+))?")


def printed_half_ulp(text: str, value: float) -> float:
    """Half a unit in the last digit `text` printed for `value`, in value's own units.
    0 when the printed number cannot be matched to the value."""
    for m in _TOKEN.finditer(text):
        x = abs(float(m.group().replace(",", "")))
        for scale in _SCALES:
            if x and math.isclose(x * scale, abs(value), rel_tol=1e-6):
                return 0.5 * 10 ** -len(m.group(1) or "") * scale
    return 0.0


def mentions_value(quote: str, claim: Claim) -> bool:
    """Does `quote` print the claim's number (at any scale, within the claim's rounding)?"""
    targets = [float(v) for v in (claim.value, claim.value_high) if v is not None]
    for t in targets:
        tol = max(printed_half_ulp(claim.text, t), 1e-9 * abs(t))
        for m in _TOKEN.finditer(quote):
            n = float(m.group().replace(",", ""))
            if any(abs(n * sc - t) <= tol or abs(n * sc + t) <= tol for sc in _SCALES):
                return True
    return False


def _prev(fy: int, fp: str, comparison: str) -> tuple[int, str] | None:
    if comparison == "yoy":
        return fy - 1, fp
    if fp == "FY":
        return None  # qoq on a fiscal year is meaningless
    return (fy - 1, "Q4") if fp == "Q1" else (fy, f"Q{int(fp[1]) - 1}")


def periods(claim: Claim) -> list[tuple[int, str]] | None:
    """The fact periods this claim needs: [target] or [target, base] for a change claim."""
    fy = claim.target_fiscal_year or claim.fiscal_year
    fp = claim.target_fiscal_period or claim.fiscal_period
    if claim.comparison is None:
        return [(fy, fp)]
    base = _prev(fy, fp, claim.comparison)
    return None if base is None else [(fy, fp), base]


def run(s: Session, claim: Claim, ctx: Ctx, lock: Lock) -> tuple[EvidenceItem, Usage]:
    ps = periods(claim)
    if ps is None:
        return _insufficient("qoq comparison on a fiscal year has no base period"), OFFLINE
    with lock:
        facts = [get_xbrl_fact(s, ctx.cik, claim.metric, fy, fp) for fy, fp in ps]
    return assess(claim, facts), OFFLINE


def _insufficient(why: str, **payload) -> EvidenceItem:
    return EvidenceItem(agent="xbrl_table", stance="insufficient", summary=why, payload=payload)


def _cite(f: Fact) -> list[Citation]:
    return [Citation(kind="xbrl_fact", ref_id=i, value=f.value, unit=f.unit) for i in f.fact_ids]


def _label(f: Fact) -> str:
    return f"{f.concept} {f.fiscal_year} {f.fiscal_period}"


def assess(claim: Claim, facts: list[Fact | None]) -> EvidenceItem:
    """Pure comparison of the claim against pre-fetched facts ([target] or [target, base])."""
    if any(f is None for f in facts):
        return _insufficient(
            f"no filed XBRL fact for {claim.metric} in the needed period(s)",
            missing=[i for i, f in enumerate(facts) if f is None],
        )
    if claim.comparison is None and claim.unit != CONCEPTS[claim.metric].unit:
        return _insufficient(f"claim unit {claim.unit} does not match {claim.metric}")
    lo = float(claim.value)
    hi = float(claim.value_high) if claim.value_high is not None else lo
    half_ulp = max(printed_half_ulp(claim.text, lo), printed_half_ulp(claim.text, hi))
    cur = facts[0]
    derived = any(f.derived for f in facts)
    cites = [c for f in facts for c in _cite(f)]

    if len(facts) == 1:
        actual = cur.value
        tol = REL_TOL * abs(actual) + half_ulp
        nearest = min(max(actual, lo), hi)
        err = nearest - actual
        deviation = err / abs(actual) if actual else err
        what = f"filed {_label(cur)}"
    else:
        base = facts[1]
        pct_change = claim.unit == "percent" and CONCEPTS[claim.metric].unit != "percent"
        if pct_change:
            if base.value == 0:
                return _insufficient("base-period value is zero; percent change undefined")
            actual = (cur.value - base.value) / abs(base.value) * 100
            tol = PP_TOL + half_ulp
        else:  # margin points, or an absolute change in dollars
            actual = cur.value - base.value
            tol = (PP_TOL if claim.unit == "percent" else REL_TOL * abs(actual)) + half_ulp
        nearest = min(max(actual, lo), hi)
        err = nearest - actual
        deviation = err / 100 if claim.unit == "percent" else (err / abs(actual) if actual else err)
        what = f"{claim.comparison} change from filed {_label(base)} to {_label(cur)}"

    ok = abs(err) <= tol
    return EvidenceItem(
        agent="xbrl_table",
        stance="supports" if ok else "contradicts",
        summary=f"Claimed value is {'within' if ok else 'outside'} tolerance of the {what}.",
        citations=cites,
        payload={
            "actual": actual, "claimed_low": lo, "claimed_high": hi, "tolerance": tol,
            "deviation": deviation, "tol_used": abs(err) / max(tol, 1e-12),
            "derived": derived, "periods": [[f.fiscal_year, f.fiscal_period] for f in facts],
        },
    )

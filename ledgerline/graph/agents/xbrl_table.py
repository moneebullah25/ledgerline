"""XBRLTable agent (FR-22 typed query). Compares a claim with filed XBRL facts only; it
never reads a number out of text.

Tolerance, level claims: |actual - claimed| <= REL_TOL * |actual| (PP_TOL for a percent
concept such as a margin) + half a unit in the last digit the claim printed. "$94.9 billion" printed one decimal of billions, so the rounding
part is 0.05e9: it is consistent with anything in [94.85e9, 94.95e9] plus 0.5% slack for
the gap between release rounding and the filed figure.

Tolerance, change claims (yoy/qoq): |actual_pct - claimed_pct| <= PP_TOL + half a unit in
the last printed digit of the percent ("grew 5%" allows 4.4..5.6 with PP_TOL=0.1).
For margin concepts (already percent) the change is the difference in points. An absolute
change in dollars allows REL_TOL * (|current| + |base|): both filed values carry rounding.

Loss claims: "net loss of $2.1 billion" is extracted as +2.1e9 while NetIncomeLoss is filed
as -2.1e9, so a positive value in a loss-phrased claim on a signed metric is compared as
negative. A loss claim against a filed profit (or the reverse) is then a contradiction.

Citations: every filed fact is cited with its own value and unit (a margin cites its USD
numerator and denominator); values the agent computed (the margin itself, the change, the
tolerance) are cited as kind="computed" so a rationale may print them.

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
_SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PCT = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s?%|\s(?:percent|percentage points?|points?|pts)\b)")
_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.(\d+))?")
_SIGNED = {"net_income", "operating_income", "eps_diluted", "free_cash_flow"}
_LOSS = re.compile(r"\b(?:net|operating)\s+loss(?:es)?\b|\bloss per\b", re.IGNORECASE)
_INCOME = re.compile(r"\b(?:income|earnings|profit)\b", re.IGNORECASE)


def printed_half_ulp(text: str, value: float, unit: str | None = None) -> float:
    """Half a unit in the last digit `text` printed for `value`, in value's own units.
    For a percent claim, numbers followed by %/points are tried first ("$5.0 billion, up
    5%" printed the 5 with no decimals). 0 when no printed number matches the value."""
    pct = [m.group(1) for m in _PCT.finditer(text)] if unit == "percent" else []
    for tok in pct + [m.group() for m in _TOKEN.finditer(text)]:
        x = abs(float(tok.replace(",", "")))
        for scale in _SCALES:
            if x and math.isclose(x * scale, abs(value), rel_tol=1e-6):
                return 0.5 * 10 ** -len(tok.partition(".")[2]) * scale
    return 0.0


def mentions_value(quote: str, claim: Claim) -> bool:
    """Does `quote` print the claim's number, in the claim's unit, within its rounding?
    Dollars must carry "$" (any scale for USD, none for per-share); percents must carry
    "%", "percent" or "points": a bare "18" in "Rule 10b-18" is not "18%"."""
    if claim.unit in ("USD", "USD/share"):
        rx, scales = _MONEY, (_SCALES if claim.unit == "USD" else (1.0,))
    elif claim.unit == "percent":
        rx, scales = _PCT, (1.0,)
    else:
        rx, scales = _TOKEN, (1.0,)
    for t in (float(v) for v in (claim.value, claim.value_high) if v is not None):
        tol = max(printed_half_ulp(claim.text, t, claim.unit), 1e-9 * abs(t))
        for m in rx.finditer(quote):
            n = float(m.group(1 if rx is not _TOKEN else 0).replace(",", ""))
            if any(abs(n * sc - abs(t)) <= tol for sc in scales):
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
    if f.components:  # each component with its own value, then the computed ratio/diff
        return [c for x in f.components for c in _cite(x)] + [
            Citation(kind="computed", ref_id=f"{f.concept}:{f.fiscal_year}{f.fiscal_period}",
                     value=f.value, unit=f.unit)]
    return [Citation(kind="xbrl_fact", ref_id=i, value=f.value, unit=f.unit) for i in f.fact_ids]


def _label(f: Fact) -> str:
    return f"{f.concept} {f.fiscal_year} {f.fiscal_period}"


def _loss_claim(claim: Claim) -> bool:
    # ponytail: a sentence naming both a loss and income ("net income of $5B vs a net loss")
    # is left unsigned; locating the phrase next to the printed value would resolve it.
    return (claim.metric in _SIGNED and bool(_LOSS.search(claim.text))
            and not _INCOME.search(claim.text))


def assess(claim: Claim, facts: list[Fact | None]) -> EvidenceItem:
    """Pure comparison of the claim against pre-fetched facts ([target] or [target, base])."""
    if any(f is None for f in facts):
        return _insufficient(
            f"no filed XBRL fact for {claim.metric} in the needed period(s)",
            missing=[i for i, f in enumerate(facts) if f is None],
        )
    unit = CONCEPTS[claim.metric].unit
    if claim.unit not in ((unit,) if claim.comparison is None else ("percent", unit)):
        return _insufficient(f"claim unit {claim.unit} does not match {claim.metric}")
    lo = float(claim.value)
    hi = float(claim.value_high) if claim.value_high is not None else lo
    if claim.comparison is None and lo > 0 and _loss_claim(claim):
        lo, hi = -hi, -lo
    half_ulp = max(printed_half_ulp(claim.text, lo, claim.unit),
                   printed_half_ulp(claim.text, hi, claim.unit))
    cur = facts[0]
    derived = any(f.derived for f in facts)
    cites = [c for f in facts for c in _cite(f)]
    ref = f"{claim.metric}:{cur.fiscal_year}{cur.fiscal_period}"

    if len(facts) == 1:
        actual = cur.value
        tol = (PP_TOL if unit == "percent" else REL_TOL * abs(actual)) + half_ulp
        nearest = min(max(actual, lo), hi)
        err = nearest - actual
        deviation = err / abs(actual) if actual else err
        what = f"filed {_label(cur)}"
    else:
        base = facts[1]
        if claim.unit == "percent" and unit != "percent":
            if base.value == 0:
                return _insufficient("base-period value is zero; percent change undefined")
            actual = (cur.value - base.value) / abs(base.value) * 100
            tol = PP_TOL + half_ulp
        elif claim.unit == "percent":  # margin points
            actual = cur.value - base.value
            tol = PP_TOL + half_ulp
        else:  # an absolute change in the concept's unit
            actual = cur.value - base.value
            tol = REL_TOL * (abs(cur.value) + abs(base.value)) + half_ulp
        nearest = min(max(actual, lo), hi)
        err = nearest - actual
        deviation = err / 100 if claim.unit == "percent" else (err / abs(actual) if actual else err)
        what = f"{claim.comparison} change from filed {_label(base)} to {_label(cur)}"
        ref += f":{claim.comparison}_change"
        cites.append(Citation(kind="computed", ref_id=ref, value=actual, unit=claim.unit))
    cites.append(Citation(kind="computed", ref_id=f"{ref}:tolerance", value=tol, unit=claim.unit))

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

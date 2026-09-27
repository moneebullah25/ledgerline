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

"Flat" prints no digit; it is read as a change that rounds to 0 at whole-percent precision,
the same half unit a printed "0%" would get (so -0.4% is flat, -0.7% is not).

Parts of a metric are not the metric: a claimed number led by "includes" / "impact of" /
"charges of" / "contributed", or followed by "of diluted earnings per share" ("the 53rd week
adds $0.30 of diluted EPS"), is a component, and comparing it with the filed total says
nothing. Such claims are `insufficient`.

Company-defined measures (free cash flow): the derived OCF - capex is one common definition,
not the company's. A match supports; a mismatch may be a different definition (META also
subtracts finance-lease principal payments, which no mapped tag carries), so it is
`insufficient`, never a contradiction.

Loss claims: extraction stores "net loss of $2.1 billion" as -2.1e9, which compares directly
with NetIncomeLoss filed as -2.1e9. Older or model-extracted claims may carry +2.1e9, so a
positive value whose nearest metric phrase is a loss phrase is compared as negative ("net loss
of $1B versus net income of $2B" signs only the $1B). A loss claim against a filed profit (or
the reverse) is then a contradiction. A change claim on a loss with a negative base ("net loss
widened 20%") compares magnitudes: -1.0 to -1.2 widened the loss by 20%.

Floors: "revenues exceeded $400 billion" / "grew more than 20%" is consistent with any filed
value above the printed one.

Basis points: "up 120 basis points" is 1.20 points, printed to a hundredth of a point.

Citations: every filed fact is cited with its own value and unit (a margin cites its USD
numerator and denominator); values the agent computed (the margin itself, the change, the
tolerance) are cited as kind="computed" so a rationale may print them.

Guidance whose target period has been reported is checked the same way: supports means
the reported value landed inside the guided value/range.

payload["deviation"] feeds severity: for levels, (nearest claimed value - actual) / |actual|;
for changes, (claimed - actual) in points / 100. It is None when the filed value is 0 (no
relative error exists; the absolute amount must not pass for one). payload["tol_used"] =
|error| / tolerance.

Implausible gaps: a dollar or per-share result more than IMPLAUSIBLE_GAP from the same-sign
filed fact is `insufficient` with payload["implausible_gap"], citing the facts and the gap in
percent so the refusal can say so. Percent concepts are exempt: a margin 13 points off is a
contradiction to rule on, not a misread scale.
"""

import math
import re
from threading import Lock

from sqlalchemy.orm import Session

from ledgerline.concepts import CONCEPTS
from ledgerline.extract.normalize import Value, parse_values
from ledgerline.graph.state import Ctx
from ledgerline.llm import OFFLINE, Usage
from ledgerline.models import Claim
from ledgerline.schemas import Citation, EvidenceItem
from ledgerline.store.xbrl_query import Fact, get_xbrl_fact

REL_TOL = 0.005
# A reported figure more than this far (relative) from the filed one is almost always a
# misread period or scope (a quarter read as the year, dividends read as cash flow), not a
# misstatement: the agent abstains instead of contradicting. Perturbed test claims move
# 15-25%, well inside it. Dollar and per-share concepts only (see module docstring).
IMPLAUSIBLE_GAP = 0.5
_GAP_UNITS = ("USD", "USD/share")
PP_TOL = 0.1
EPS_HALF_CENT = 0.005
_SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PCT = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s?%|\s(?:percent|percentage points?|points?|pts)\b)")
_BPS = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)[\s-]?(?:basis[\s-]points?|bps)\b",
                  re.IGNORECASE)
_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.(\d+))?")
_SIGNED = {"net_income", "operating_income", "eps_diluted", "free_cash_flow"}
_LOSS = re.compile(r"\b(?:net|operating)\s+loss(?:es)?\b|\bloss (?:per|from operations)\b",
                   re.IGNORECASE)
_INCOME = re.compile(r"\b(?:income|earnings|profit)\b", re.IGNORECASE)
_FLAT = re.compile(r"\bflat\b", re.IGNORECASE)
_WORDS = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()))
# ponytail: cue words only; "revenue including acquisitions was $5B" also reads as a part.
_PART_BEFORE = re.compile(r"\b(?:includ(?:e|es|ed|ing)|impacts?|benefits?|charges?|effects?"
                          r"|contribut\w*)\b[^,;$%]*$", re.IGNORECASE)
_PART_AFTER = re.compile(r"^\s*of\s+(?:\w+\s+){0,2}(?:earnings per share|eps)\b", re.IGNORECASE)
_COMPANY_DEFINED = {"free_cash_flow"}
# "revenues exceeded $400 billion", "grew more than 20%": a floor, not a point estimate.
_FLOOR = re.compile(r"\b(?:exceed(?:ed|s|ing)?|more than|over|above|surpass(?:ed|es|ing)?|topp(?:ed|ing)"
                    r"|in excess of)\s*$", re.IGNORECASE)


def printed_half_ulp(text: str, value: float, unit: str | None = None) -> float:
    """Half a unit in the last digit `text` printed for `value`, in value's own units.
    For a percent claim, numbers followed by %/points are tried first ("$5.0 billion, up
    5%" printed the 5 with no decimals). 0 when no printed number matches the value."""
    toks = [(m.group(), 1.0) for m in _TOKEN.finditer(text)]
    if unit == "percent":  # a basis point is a hundredth of a point
        toks = ([(m.group(1), 0.01) for m in _BPS.finditer(text)]
                + [(m.group(1), 1.0) for m in _PCT.finditer(text)] + toks)
    for tok, per in toks:
        x = abs(float(tok.replace(",", ""))) * per
        for scale in _SCALES:
            if x and math.isclose(x * scale, abs(value), rel_tol=1e-6):
                return 0.5 * 10 ** -len(tok.partition(".")[2]) * per * scale
    if unit == "percent" and value == 0 and _FLAT.search(text):
        return 0.5
    # "an increase of two percent": a spelled-out number prints whole units
    if unit == "percent" and abs(value) in _WORDS and re.search(
            rf"\b{_WORDS[abs(value)]}\b", text, re.IGNORECASE):
        return 0.5
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


def _claimed_spans(claim: Claim) -> list[Value]:
    """The printed values in the claim text that are the claim's value (or range end)."""
    claimed = [abs(float(x)) for x in (claim.value, claim.value_high) if x is not None]
    return [v for v in parse_values(claim.text)
            if any(x is not None and math.isclose(abs(x), c, rel_tol=1e-6)
                   for x in (v.value, v.value_high) for c in claimed)]


def _loss_claim(claim: Claim) -> bool:
    """Is the metric phrase nearest the claimed value a loss phrase ("net loss of $2.1B")?
    Only that phrase counts: "net loss of $1B versus net income of $2B" is a loss claim."""
    if claim.metric not in _SIGNED:
        return False
    spans = _claimed_spans(claim)
    if not spans:  # value not found in the text: fall back to the whole sentence
        return bool(_LOSS.search(claim.text)) and not _INCOME.search(claim.text)
    v = spans[0]
    phrases = [(v.start - m.end() if m.end() <= v.start else m.start() - v.end, rx is _LOSS)
               for rx in (_LOSS, _INCOME) for m in rx.finditer(claim.text)
               if m.end() <= v.start or m.start() >= v.end]
    return min(phrases)[1] if phrases else False


def _is_part(claim: Claim) -> bool:
    """Is the claimed number a component of the metric rather than its total?"""
    return any(_PART_BEFORE.search(claim.text[:v.start]) or _PART_AFTER.match(claim.text[v.end:])
               for v in _claimed_spans(claim))


def assess(claim: Claim, facts: list[Fact | None]) -> EvidenceItem:
    """Pure comparison of the claim against pre-fetched facts ([target] or [target, base])."""
    if _is_part(claim):
        return _insufficient(f"the claimed figure is a component of {claim.metric}, "
                             "not the filed total")
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
    loss = _loss_claim(claim)
    if claim.comparison is None and lo > 0 and loss:
        lo, hi = -hi, -lo
    # A floor: anything above it is consistent; the tolerance still applies below it.
    floor = lo >= 0 and not loss and any(_FLOOR.search(claim.text[:v.start])
                                         for v in _claimed_spans(claim))
    half_ulp = max(printed_half_ulp(claim.text, lo, claim.unit),
                   printed_half_ulp(claim.text, hi, claim.unit))
    cur = facts[0]
    derived = any(f.derived for f in facts)
    cites = [c for f in facts for c in _cite(f)]
    ref = f"{claim.metric}:{cur.fiscal_year}{cur.fiscal_period}"

    if len(facts) == 1:
        actual = cur.value
        tol = (PP_TOL if unit == "percent" else REL_TOL * abs(actual)) + half_ulp
        nearest = min(max(actual, lo), math.inf if floor else hi)
        err = nearest - actual
        deviation = err / abs(actual) if actual else None
        what = f"filed {_label(cur)}"
    else:
        base = facts[1]
        cur_v, base_v = cur.value, base.value
        if loss and base_v < 0:  # "net loss widened 20%": the change in the loss's size
            cur_v, base_v = -cur_v, -base_v
        if claim.unit == "percent" and unit != "percent":
            if base.value == 0:
                return _insufficient("base-period value is zero; percent change undefined")
            actual = (cur_v - base_v) / abs(base_v) * 100
            tol = PP_TOL + half_ulp
            if unit == "USD/share":
                # EPS is filed to the cent; the company's percent comes from unrounded
                # figures ($1.70 vs $1.33 is +27.8%, PepsiCo prints 27%). Half a cent on each.
                tol += 100 * EPS_HALF_CENT * (abs(cur_v) + abs(base_v)) / base_v ** 2
        elif claim.unit == "percent":  # margin points
            actual = cur.value - base.value
            tol = PP_TOL + half_ulp
        else:  # an absolute change in the concept's unit
            actual = cur_v - base_v
            tol = REL_TOL * (abs(cur.value) + abs(base.value)) + half_ulp
        nearest = min(max(actual, lo), math.inf if floor else hi)
        err = nearest - actual
        deviation = err / 100 if claim.unit == "percent" else (err / abs(actual) if actual else None)
        what = f"{claim.comparison} change from filed {_label(base)} to {_label(cur)}"
        ref += f":{claim.comparison}_change"
        cites.append(Citation(kind="computed", ref_id=ref, value=actual, unit=claim.unit))
    cites.append(Citation(kind="computed", ref_id=f"{ref}:tolerance", value=tol, unit=claim.unit))

    ok = abs(err) <= tol
    # Same sign only: a loss claimed against a filed profit (or the reverse) is a real
    # contradiction, not a misread scale.
    if (not ok and claim.claim_type == "reported_result" and claim.comparison is None
            and unit in _GAP_UNITS and lo * actual > 0 and abs(deviation) > IMPLAUSIBLE_GAP):
        gap = abs(deviation) * 100
        return EvidenceItem(
            agent="xbrl_table", stance="insufficient",
            summary=(f"claimed value is {gap:.0f}% from the {what}: more likely a period or "
                     "scope misreading than a misstatement"),
            citations=[*cites[:-1],  # the facts, minus the tolerance
                       Citation(kind="computed", ref_id=f"{ref}:gap_pct", value=gap,
                                unit="percent")],
            payload={"implausible_gap": True, "gap_pct": gap, "actual": actual,
                     "deviation": deviation})
    if not ok and claim.metric in _COMPANY_DEFINED:
        return _insufficient(f"{claim.metric} is company-defined; the derived definition does "
                             "not match and no filed fact carries the company's", actual=actual)
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

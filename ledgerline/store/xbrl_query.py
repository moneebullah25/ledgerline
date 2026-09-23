"""Typed fact lookup: the anti-hallucination primitive (FR-22).

A number the verifier compares a claim against comes from here or not at all. Anything
missing (no mapped tag, no fact for the period, a zero denominator) returns None (R8).
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from ledgerline.concepts import CONCEPTS
from ledgerline.ingest.xbrl import SPANS
from ledgerline.models import Company, XbrlFact


@dataclass(frozen=True)
class Fact:
    value: float
    unit: str
    fact_ids: tuple[int, ...]
    concept: str
    fiscal_year: int
    fiscal_period: str
    period_end: date
    derived: bool
    source_url: str


def _period_filter(fiscal_period: str):
    lo, hi = SPANS["Y" if fiscal_period == "FY" else "Q"]
    # Balance-sheet instants carry no duration; the year-end instant answers Q4 as well as FY.
    instant_fp = "FY" if fiscal_period in ("Q4", "FY") else fiscal_period
    return or_(
        and_(XbrlFact.fiscal_period == fiscal_period, XbrlFact.duration_days.between(lo, hi)),
        and_(XbrlFact.fiscal_period == instant_fp, XbrlFact.duration_days.is_(None)),
    )


def get_xbrl_fact(s: Session, cik: str, concept: str, fiscal_year: int,
                  fiscal_period: str) -> Fact | None:
    c = CONCEPTS.get(concept)
    if c is None:
        return None
    if c.ratio_of or c.diff_of:
        a_name, b_name = c.ratio_of or c.diff_of
        a = get_xbrl_fact(s, cik, a_name, fiscal_year, fiscal_period)
        b = get_xbrl_fact(s, cik, b_name, fiscal_year, fiscal_period)
        if a is None or b is None or (c.ratio_of and b.value == 0):
            return None
        value = a.value / b.value * 100 if c.ratio_of else a.value - b.value
        return Fact(value, c.unit, a.fact_ids + b.fact_ids, concept, fiscal_year,
                    fiscal_period, a.period_end, True, a.source_url)

    for tag in c.tags:  # preference order: the first tag with a value for the period wins
        row = s.scalars(
            select(XbrlFact).join(Company, Company.id == XbrlFact.company_id)
            .where(Company.cik == cik.zfill(10), XbrlFact.tag == tag, XbrlFact.unit == c.unit,
                   XbrlFact.fiscal_year == fiscal_year, _period_filter(fiscal_period))
            # As-filed beats derived; among duplicates the earliest original filing wins
            # over later amendments (what the market saw at the time of the claim).
            .order_by(XbrlFact.derived, XbrlFact.form.like("%/A"), XbrlFact.filed, XbrlFact.id)
            .limit(1)
        ).first()
        if row is not None:
            return Fact(float(row.value), c.unit, (row.id,), concept, fiscal_year,
                        fiscal_period, row.period_end, row.derived, row.source_url)
    return None


def get_series(s: Session, cik: str, concept: str, fiscal_period_kind: str = "Q") -> list[Fact]:
    """Quarterly (kind "Q") or annual (kind "FY") history, oldest first."""
    fps = {"Q": ("Q1", "Q2", "Q3", "Q4"), "FY": ("FY",)}[fiscal_period_kind]
    periods = s.execute(
        select(XbrlFact.fiscal_year, XbrlFact.fiscal_period).distinct()
        .join(Company, Company.id == XbrlFact.company_id)
        .where(Company.cik == cik.zfill(10), XbrlFact.fiscal_period.in_(fps))
    ).all()
    facts = (get_xbrl_fact(s, cik, concept, fy, fp) for fy, fp in periods)
    return sorted((f for f in facts if f is not None), key=lambda f: (f.period_end, f.fiscal_period))

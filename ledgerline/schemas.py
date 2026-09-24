"""§22 contracts shared by extraction, the verification graph, the report, and the API.

The citation-integrity rule (FR-45 / NFR-8) lives here so every producer of a verdict is
held to the same check: a number a reader sees must come from cited evidence.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

FiscalPeriod = Literal["Q1", "Q2", "Q3", "Q4", "FY"]
ClaimType = Literal["reported_result", "comparison", "guidance", "qualitative"]
Unit = Literal["USD", "USD/share", "percent", "count", "ratio"]
Stance = Literal["supports", "contradicts", "insufficient"]
Label = Literal["supported", "contradicted", "unverifiable"]
AgentName = Literal["filing_rag", "xbrl_table", "forecast", "market_reaction"]


class ClaimDraft(BaseModel):
    """Output of extraction (§22.1), before it is persisted as a claim row."""

    text: str = Field(min_length=1)
    span_start: int = Field(ge=0)
    span_end: int
    claim_type: ClaimType
    metric: str | None = None  # canonical concept, see store.xbrl_query.CONCEPTS
    value: float | None = None
    value_high: float | None = None
    unit: Unit | None = None
    target_fiscal_year: int | None = None
    target_fiscal_period: FiscalPeriod | None = None
    comparison: Literal["yoy", "qoq"] | None = None
    hedging_score: float = Field(0.0, ge=0, le=1)
    speaker_role: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "ClaimDraft":
        if self.span_end <= self.span_start:
            raise ValueError("span_end must be after span_start")
        if self.value is not None and self.unit is None:
            raise ValueError("a numeric claim needs a unit")
        if self.value_high is not None and (self.value is None or self.value_high < self.value):
            raise ValueError("value_high needs value and must be >= value")
        if self.claim_type == "comparison" and self.comparison is None:
            raise ValueError("a comparison claim must say yoy or qoq")
        return self


class Citation(BaseModel):
    # "computed": a value the verifier derived from cited facts (a margin, a change); its
    # ref_id names the computation and the component facts are cited alongside it.
    kind: Literal["xbrl_fact", "chunk", "price", "forecast", "segment", "computed"]
    ref_id: int | str
    quote: str | None = None  # exact text span for chunk/segment citations
    value: float | None = None  # the number this citation contributes, if any
    unit: str | None = None


class EvidenceItem(BaseModel):
    """What one evidence agent returns (§22.2). Agents never emit a verdict."""

    agent: AgentName
    stance: Stance
    summary: str
    citations: list[Citation] = []
    payload: dict = {}

    @model_validator(mode="after")
    def _cited(self) -> "EvidenceItem":
        if self.stance != "insufficient" and not self.citations:
            raise ValueError(f"{self.agent}: a {self.stance} stance needs at least one citation")
        return self


class CriticOutput(BaseModel):
    """The adversarial critic argues the other side (FR-42). It has no label field on purpose."""

    counter_stance: Stance
    argument: str
    weakest_evidence: str | None = None


class VerdictDraft(BaseModel):
    """Adjudicator output (§22.4)."""

    label: Label
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    citations: list[Citation]

    @model_validator(mode="after")
    def _grounded(self) -> "VerdictDraft":
        if self.label != "unverifiable" and not self.citations:
            raise ValueError("supported/contradicted verdicts must cite evidence")
        return self


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
# A number next to one of these is an amount, never an ordinal or a year. A hyphen joins only
# a spelled-out scale ("25-billion"): "10-K" is a form name.
_UNIT_BEFORE = re.compile(r"(?:[$€£]|\busd)\s*$", re.IGNORECASE)
_UNIT_AFTER = re.compile(
    r"(?:\s*|-(?![kmb]\b))(?:%|(?:percent|per cent|percentage|points?|pp|bps|basis|thousand|million|billion"
    r"|trillion|usd|bn|mn|[kmb])\b)", re.IGNORECASE)
_SMALL = 32


def numbers_in(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM.findall(text)]


def _renderings(v: float) -> set[float]:
    """The ways a cited value can legitimately appear in prose: raw, or in thousands,
    millions, or billions, rounded to what a writer would print. A negative value may also
    be printed as its magnitude ("net loss of $2.1 billion")."""
    out = set()
    out |= {v, abs(v)}  # a value copied at full precision
    for x0 in {v, abs(v)}:
        for scale in (1, 1e3, 1e6, 1e9):
            x = x0 / scale
            out |= {round(x, 0), round(x, 1), round(x, 2), round(x, 3)}
    return out


def _bare_ordinal_or_year(n: float, text: str, start: int, end: int) -> bool:
    """Integers < 32 ("Q3", "4 agents") and fiscal years pass without a citation, but only
    when no currency, percent, or scale word marks them as an amount ("$25 billion")."""
    if not n.is_integer() or not (abs(n) < _SMALL or 1990 <= n <= 2100):
        return False
    return not (_UNIT_BEFORE.search(text[:start]) or _UNIT_AFTER.match(text, end))


def uncited_numbers(text: str, citations: list[Citation], claim_text: str = "") -> list[float]:
    """Numbers in `text` that no citation (or the claim being checked) accounts for.

    An empty list means the text passes citation integrity (FR-45). The claim's own
    numbers may be restated; everything else must trace to cited evidence.
    """
    allowed: set[float] = set(numbers_in(claim_text))
    for c in citations:
        if c.value is not None:
            allowed |= _renderings(c.value)
        if c.quote:
            allowed |= set(numbers_in(c.quote))
    out = []
    for m in _NUM.finditer(text):
        n = float(m.group().replace(",", ""))
        if not _bare_ordinal_or_year(n, text, m.start(), m.end()) and not any(
                abs(n - a) <= 1e-9 * max(1.0, abs(a)) for a in allowed):
            out.append(n)
    return out

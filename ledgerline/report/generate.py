"""The report view (FR-50): claims ranked by severity, every citation resolved to a source a
reader can open. Numbers shown come from the cited rows themselves (NFR-8), and each
finding carries the result of the citation-integrity check on its rationale."""

import textwrap
import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline.models import (
    CallAudio,
    Chunk,
    Claim,
    Company,
    Filing,
    MarketPrice,
    Report,
    Segment,
    Verdict,
    XbrlFact,
)
from ledgerline.schemas import Citation, uncited_numbers


class Source(BaseModel):
    kind: str
    ref_id: int | str
    label: str  # human-readable: what the row is and when it is from
    url: str | None = None
    quote: str | None = None
    value: float | None = None
    unit: str | None = None


class Finding(BaseModel):
    claim_id: int
    text: str
    claim_type: str
    metric: str | None
    speaker_role: str | None
    hedging_score: float
    label: str
    confidence: float
    severity: float
    refused: bool
    rationale: str
    sources: list[Source]
    uncited_numbers: list[float]  # empty = the rationale passes FR-45


class ReportView(BaseModel):
    id: uuid.UUID
    ticker: str
    fiscal_year: int
    fiscal_period: str
    status: str
    created_at: datetime
    finished_at: datetime | None
    latency_s: float | None
    cost_usd: float | None
    error: str | None
    claims: list[Finding]


def filing_index_url(cik: str, accession_no: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no.replace('-', '')}/"


def _ids(cites: list[Citation], kind: str) -> set[int]:
    return {int(c.ref_id) for c in cites if c.kind == kind and str(c.ref_id).isdigit()}


def _resolve(s: Session, cik: str, cites: list[Citation]) -> dict[tuple[str, str], Source]:
    """One query per citation kind, whatever the number of claims (keeps NFR-2 flat)."""
    out: dict[tuple[str, str], Source] = {}
    if ids := _ids(cites, "xbrl_fact"):
        for f in s.scalars(select(XbrlFact).where(XbrlFact.id.in_(ids))):
            out["xbrl_fact", str(f.id)] = Source(
                kind="xbrl_fact", ref_id=f.id, value=float(f.value), unit=f.unit,
                label=f"{f.tag}, FY{f.fiscal_year} {f.fiscal_period} "
                      f"(period ending {f.period_end}), {f.form} `{f.accession_no}` filed {f.filed}"
                      + (", derived" if f.derived else ""),
                url=filing_index_url(cik, f.accession_no),
            )
    if ids := _ids(cites, "chunk"):
        rows = s.execute(select(Chunk, Filing).join(Filing).where(Chunk.id.in_(ids)))
        for c, f in rows:
            out["chunk", str(c.id)] = Source(
                kind="chunk", ref_id=c.id, url=f.doc_url, quote=c.text,
                label=f"{f.form_type} filed {f.filed_at}, {c.section_path}",
            )
    if ids := _ids(cites, "price"):
        for p in s.scalars(select(MarketPrice).where(MarketPrice.id.in_(ids))):
            out["price", str(p.id)] = Source(
                kind="price", ref_id=p.id, value=p.close, unit="USD", url=p.source_url,
                label=f"{p.ticker} close on {p.day}",
            )
    if ids := _ids(cites, "segment"):
        rows = s.execute(select(Segment, CallAudio).join(CallAudio).where(Segment.id.in_(ids)))
        for g, a in rows:
            out["segment", str(g.id)] = Source(
                kind="segment", ref_id=g.id, quote=g.text, url=a.source_url,
                label=f"{a.kind} on {a.event_date}, {g.role} at `{g.t_start:.1f}s-{g.t_end:.1f}s`",
            )
    return out


def _source(c: Citation, resolved: dict[tuple[str, str], Source]) -> Source:
    base = resolved.get((c.kind, str(c.ref_id)))
    if base is None:  # forecast outputs, or a row that no longer exists: show what was cited
        return Source(kind=c.kind, ref_id=c.ref_id, label=f"{c.kind} {c.ref_id}",
                      quote=c.quote, value=c.value, unit=c.unit)
    # The agent's exact quote beats the whole chunk; the row's value beats the agent's copy.
    return base.model_copy(update={"quote": c.quote or base.quote})


def build_report(s: Session, report_id: uuid.UUID) -> ReportView | None:
    row = s.execute(
        select(Report, Company).join(Company).where(Report.id == report_id)
    ).first()
    if row is None:
        return None
    report, company = row
    pairs = s.execute(
        select(Verdict, Claim).join(Claim, Claim.id == Verdict.claim_id)
        .where(Verdict.report_id == report_id)
        .order_by(Verdict.severity.desc(), Verdict.confidence.desc(), Claim.id)
    ).all()
    cites = {v.id: [Citation.model_validate(c) for c in v.citations] for v, _ in pairs}
    resolved = _resolve(s, company.cik, [c for cs in cites.values() for c in cs])

    findings = []
    for v, claim in pairs:
        sources = [_source(c, resolved) for c in cites[v.id]]
        # Check against what the reader sees: resolved row values, not only the agent's copy.
        shown = cites[v.id] + [Citation(kind=x.kind, ref_id=x.ref_id, value=x.value,
                                        quote=x.quote) for x in sources]
        findings.append(Finding(
            claim_id=claim.id, text=claim.text, claim_type=claim.claim_type,
            metric=claim.metric, speaker_role=claim.speaker_role,
            hedging_score=claim.hedging_score, label=v.label, confidence=v.confidence,
            severity=v.severity, refused=v.refused, rationale=v.rationale, sources=sources,
            uncited_numbers=uncited_numbers(v.rationale, shown, claim.text),
        ))
    return ReportView(
        id=report.id, ticker=company.ticker, fiscal_year=report.fiscal_year,
        fiscal_period=report.fiscal_period, status=report.status,
        created_at=report.created_at, finished_at=report.finished_at,
        latency_s=report.latency_s, cost_usd=report.cost_usd, error=report.error,
        claims=findings,
    )


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if v.is_integer() else f"{v:,}"


def to_markdown(view: ReportView) -> str:
    """Scores and identifiers are in backticks: they are metadata about the verdict, not
    financial figures, and the citation-integrity check skips code spans."""
    lines = [
        f"# {view.ticker} FY{view.fiscal_year} {view.fiscal_period}",
        "",
        (f"status `{view.status}` · claims `{len(view.claims)}` · "
         f"latency `{view.latency_s}` s · cost `{view.cost_usd}` USD"),
    ]
    if view.error:
        lines += ["", f"error: `{view.error}`"]
    for rank, f in enumerate(view.claims, 1):
        lines += [
            "",
            (f"## {rank}. {f.label}{' (refused: no evidence)' if f.refused else ''} "
             f"`confidence {f.confidence:.2f}` `severity {f.severity:.2f}`"),
            "",
            f"> {f.text}",
            "",
            f.rationale,
            "",
        ]
        for src in f.sources:
            line = f"- {src.label}"
            if src.value is not None:
                line += f": {_fmt(src.value)} {src.unit or ''}".rstrip()
            if src.quote:
                line += f' "{textwrap.shorten(src.quote, 300)}"'
            if src.url:
                line += f" [`source`]({src.url})"
            lines.append(line)
        if f.uncited_numbers:
            lines.append(f"- WARNING uncited numbers in rationale: `{f.uncited_numbers}`")
    return "\n".join(lines) + "\n"

"""The §15 REST API. `uvicorn ledgerline.api.main:app`

Endpoints are plain `def`: SQLAlchemy here is synchronous, and FastAPI runs `def` endpoints
in its threadpool so a slow query does not block the event loop. POST /v1/reports only
enqueues a row; the worker (ledgerline.worker) runs the pipeline.
"""

import base64
import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text, tuple_
from sqlalchemy.orm import Session

from ledgerline.db import engine
from ledgerline.models import Claim, Company, EvalRun, Evidence, Report, Span, Verdict
from ledgerline.report.generate import ReportView, build_report

app = FastAPI(title="Ledgerline", version="1.0.0")


def get_session() -> Iterator[Session]:
    # Endpoints that write commit explicitly, so nothing depends on when FastAPI runs the
    # teardown of a yield dependency relative to sending the response.
    with Session(engine(), expire_on_commit=False) as s:
        yield s


DB = Annotated[Session, Depends(get_session)]
Limit = Annotated[int, Query(ge=1, le=100)]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ReportRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z.\-]+$")
    fiscal_year: int = Field(ge=2000, le=2100)
    fiscal_period: Literal["Q1", "Q2", "Q3", "Q4", "FY"]


class Job(BaseModel):
    job_id: uuid.UUID


class ReportSummary(ORM):
    id: uuid.UUID
    ticker: str
    fiscal_year: int
    fiscal_period: str
    status: str
    created_at: datetime
    finished_at: datetime | None
    latency_s: float | None
    cost_usd: float | None


class ClaimOut(ORM):
    id: int
    company_id: int
    fiscal_year: int
    fiscal_period: str
    filing_id: int | None
    segment_id: int | None
    span_start: int
    span_end: int
    text: str
    claim_type: str
    metric: str | None
    value: float | None
    value_high: float | None
    unit: str | None
    target_fiscal_year: int | None
    target_fiscal_period: str | None
    comparison: str | None
    hedging_score: float
    speaker_role: str | None
    linked_claim_id: int | None
    extractor_version: str
    created_at: datetime


class EvidenceOut(ORM):
    id: int
    report_id: uuid.UUID
    agent: str
    stance: str
    summary: str
    citations: list[dict]
    payload: dict
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    latency_ms: int
    trace_id: uuid.UUID


class VerdictOut(ORM):
    id: int
    report_id: uuid.UUID
    label: str
    confidence: float
    rationale: str
    citations: list[dict]
    severity: float
    refused: bool
    prompt_version: str
    model: str | None
    cost_usd: float
    trace_id: uuid.UUID


class SpanOut(ORM):
    id: uuid.UUID
    trace_id: uuid.UUID
    parent_id: uuid.UUID | None
    name: str
    started_at: datetime
    ended_at: datetime
    attrs: dict


class ClaimDetail(BaseModel):
    claim: ClaimOut
    verdict: VerdictOut | None  # the most recent report's verdict
    evidence: list[EvidenceOut]  # everything that verdict's run collected, critic included
    spans: list[SpanOut]  # the full trace behind the verdict


class EvalRunOut(ORM):
    id: int
    created_at: datetime
    git_sha: str | None
    layer: str
    metrics: dict
    details: dict


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None


# Keyset pagination on (created_at, id), newest first. The cursor is opaque to clients.
def _encode(created_at: datetime, id_: object) -> str:
    return base64.urlsafe_b64encode(f"{created_at.isoformat()}|{id_}".encode()).decode()


def _decode(cursor: str) -> tuple[datetime, str]:
    try:
        ts, id_ = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        return datetime.fromisoformat(ts), id_
    except ValueError:  # covers binascii.Error and UnicodeDecodeError
        raise HTTPException(422, "invalid cursor") from None


def _company(s: Session, ticker: str) -> Company:
    company = s.scalar(select(Company).where(Company.ticker == ticker.upper()))
    if company is None:
        raise HTTPException(404, f"unknown ticker {ticker}")
    return company


@app.get("/healthz")
def healthz(s: DB) -> dict:
    s.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/v1/reports", status_code=202)
def create_report(body: ReportRequest, s: DB) -> Job:
    company = _company(s, body.ticker)
    report = Report(company_id=company.id, fiscal_year=body.fiscal_year,
                    fiscal_period=body.fiscal_period, status="queued")
    s.add(report)
    s.commit()
    return Job(job_id=report.id)


@app.get("/v1/reports/{job_id}")
def get_report(job_id: uuid.UUID, s: DB) -> ReportView:
    view = build_report(s, job_id)
    if view is None:
        raise HTTPException(404, "report not found")
    return view


@app.get("/v1/reports")
def list_reports(s: DB, limit: Limit = 20, cursor: str | None = None) -> Page[ReportSummary]:
    q = select(Report, Company.ticker).join(Company)
    if cursor:
        ts, id_ = _decode(cursor)
        try:
            key = uuid.UUID(id_)
        except ValueError:
            raise HTTPException(422, "invalid cursor") from None
        q = q.where(tuple_(Report.created_at, Report.id) < tuple_(ts, key))
    rows = s.execute(q.order_by(Report.created_at.desc(), Report.id.desc()).limit(limit + 1)).all()
    fields = ReportSummary.model_fields.keys() - {"ticker"}
    items = [ReportSummary(ticker=t, **{f: getattr(r, f) for f in fields}) for r, t in rows[:limit]]
    last = rows[limit - 1][0] if len(rows) > limit else None
    return Page(items=items, next_cursor=_encode(last.created_at, last.id) if last else None)


@app.get("/v1/claims/{claim_id}")
def get_claim(claim_id: int, s: DB) -> ClaimDetail:
    claim = s.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(404, "claim not found")
    verdict = s.scalar(select(Verdict).where(Verdict.claim_id == claim_id)
                       .order_by(Verdict.id.desc()).limit(1))
    evidence, spans = [], []
    if verdict is not None:
        evidence = s.scalars(select(Evidence).where(
            Evidence.claim_id == claim_id, Evidence.report_id == verdict.report_id,
        ).order_by(Evidence.id)).all()
        spans = s.scalars(select(Span).where(Span.trace_id == verdict.trace_id)
                          .order_by(Span.started_at)).all()
    return ClaimDetail(
        claim=ClaimOut.model_validate(claim),
        verdict=VerdictOut.model_validate(verdict) if verdict else None,
        evidence=[EvidenceOut.model_validate(e) for e in evidence],
        spans=[SpanOut.model_validate(x) for x in spans],
    )


@app.get("/v1/companies/{ticker}/claims")
def company_claims(ticker: str, s: DB, limit: Limit = 50,
                   cursor: str | None = None) -> Page[ClaimOut]:
    company = _company(s, ticker)
    q = select(Claim).where(Claim.company_id == company.id)
    if cursor:
        ts, id_ = _decode(cursor)
        if not id_.isdigit():
            raise HTTPException(422, "invalid cursor")
        q = q.where(tuple_(Claim.created_at, Claim.id) < tuple_(ts, int(id_)))
    rows = s.scalars(q.order_by(Claim.created_at.desc(), Claim.id.desc()).limit(limit + 1)).all()
    last = rows[limit - 1] if len(rows) > limit else None
    return Page(items=[ClaimOut.model_validate(c) for c in rows[:limit]],
                next_cursor=_encode(last.created_at, last.id) if last else None)


@app.get("/v1/evals/latest")
def evals_latest(s: DB) -> list[EvalRunOut]:
    """Public on purpose: the eval numbers are the product's credibility (§11, §20)."""
    runs = s.scalars(
        select(EvalRun).distinct(EvalRun.layer)
        .order_by(EvalRun.layer, EvalRun.created_at.desc(), EvalRun.id.desc())
    ).all()
    return [EvalRunOut.model_validate(r) for r in runs]

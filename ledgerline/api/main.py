"""The §15 REST API. `uvicorn ledgerline.api.main:app`

Endpoints are plain `def`: SQLAlchemy here is synchronous, and FastAPI runs `def` endpoints
in its threadpool so a slow query does not block the event loop. POST /v1/reports only
enqueues a row; the worker (ledgerline.worker) runs the pipeline.

POST /v1/reports is the only endpoint that spends money (LLM calls), so it needs the
`X-API-Key` header to match LEDGERLINE_API_KEY and is disabled when that is unset. Reads
are public. Every 422 has FastAPI's list shape: {"detail": [{"loc", "msg", "type"}]}.
"""

import base64
import hmac
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import func, select, text, tuple_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ledgerline.config import settings
from ledgerline.db import engine
from ledgerline.models import Claim, Company, EvalRun, Evidence, Report, Span, Verdict
from ledgerline.pipeline import FY_UNSUPPORTED
from ledgerline.report.generate import ReportView, build_report

# Backlog cap: beyond it POST answers 429 instead of growing the queue.
# ponytail: soft cap - the count is not under a lock shared across quarters, so concurrent POSTs
# for different quarters can overshoot it by a few; a global advisory lock makes it hard.
MAX_QUEUED = 50
CURSOR_ID_MAX_LEN = 36  # a UUID; claim ids are far shorter
INT4_MAX = 2**31 - 1  # claim.id is a Postgres integer

app = FastAPI(title="Ledgerline", version="1.0.0")


def get_session() -> Iterator[Session]:
    # Endpoints that write commit explicitly, so nothing depends on when FastAPI runs the
    # teardown of a yield dependency relative to sending the response.
    with Session(engine(), expire_on_commit=False) as s:
        yield s


DB = Annotated[Session, Depends(get_session)]
Limit = Annotated[int, Query(ge=1, le=100)]
ClaimId = Annotated[int, Path(ge=1, le=INT4_MAX)]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ReportRequest(BaseModel):
    model_config = ConfigDict(strict=True)  # "2025" is not a year, "yes" is not a bool

    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z.\-]+$")
    fiscal_year: int = Field(ge=2000, le=2100)
    fiscal_period: Literal["Q1", "Q2", "Q3", "Q4", "FY"]
    rerun: bool = False  # recompute even if a done report exists (a queued/running one is reused)

    @field_validator("fiscal_period")
    @classmethod
    def _no_fy(cls, v: str) -> str:
        if v == "FY":
            raise ValueError(FY_UNSUPPORTED)
        return v

    @model_validator(mode="after")
    def _not_future(self) -> "ReportRequest":
        # Fiscal years run up to a year ahead of the calendar (FY Y can end as early as
        # January of Y), so quarter n of FY Y cannot end before 3n months after 1 January Y-1.
        # Only a quarter that cannot have ended yet is rejected.
        n = int(self.fiscal_period[1])
        earliest_end = date(self.fiscal_year - 1 + n // 4, 3 * n % 12 + 1, 1)
        if earliest_end > datetime.now(UTC).date():
            raise ValueError(f"FY{self.fiscal_year} {self.fiscal_period} has not ended yet")
        return self


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
    cache_write_tokens: int
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
    verdict: VerdictOut | None  # from the latest finished (done) report; None if none has one
    evidence: list[EvidenceOut]  # everything that verdict's run collected, critic included
    spans: list[SpanOut]  # the full trace behind the verdict


class EvalRunOut(ORM):
    id: int
    created_at: datetime
    git_sha: str | None
    layer: str
    name: str
    metrics: dict
    details: dict


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None


# Keyset pagination on (created_at, id), newest first. The cursor is opaque to clients.
def _encode(created_at: datetime, id_: object) -> str:
    return base64.urlsafe_b64encode(f"{created_at.isoformat()}|{id_}".encode()).decode()


def _invalid(loc: tuple, msg: str, value: object) -> RequestValidationError:
    return RequestValidationError([{"type": "value_error", "loc": loc, "msg": msg,
                                    "input": value}])


def _decode(cursor: str) -> tuple[datetime, str]:
    try:
        ts, id_ = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        if not (id_.isascii() and 0 < len(id_) <= CURSOR_ID_MAX_LEN):
            raise ValueError(id_)
        return datetime.fromisoformat(ts), id_
    except ValueError:  # covers binascii.Error and UnicodeDecodeError
        raise _invalid(("query", "cursor"), "invalid cursor", cursor) from None


def _company(s: Session, ticker: str) -> Company:
    company = s.scalar(select(Company).where(Company.ticker == ticker.upper()))
    if company is None:
        raise HTTPException(404, f"unknown ticker {ticker}")
    return company


@app.get("/healthz", response_model=None,
         responses={503: {"description": "database unreachable"}})
def healthz(s: DB) -> dict | JSONResponse:
    try:
        s.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse({"status": "unavailable", "database": "unreachable"}, 503)
    return {"status": "ok"}


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    # Through Settings, so a key in .env works for a local uvicorn as well as under compose.
    secret = settings().api_key
    expected = secret.get_secret_value() if secret else ""
    if not expected:
        raise HTTPException(503, "report creation is disabled: LEDGERLINE_API_KEY is not set")
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "missing or invalid X-API-Key",
                            headers={"WWW-Authenticate": "X-API-Key"})


@app.post("/v1/reports", status_code=202, dependencies=[Depends(require_api_key)])
def create_report(body: ReportRequest, s: DB, response: Response) -> Job:
    """Idempotent per (company, quarter): an existing queued, running or done report is
    returned instead of a new one. A failed report is re-run by POSTing again; a done one
    only with `rerun: true`."""
    company = _company(s, body.ticker)
    key = (company.id, body.fiscal_year, body.fiscal_period)
    # Serialise POSTs for the same quarter so two concurrent requests cannot both insert.
    s.execute(select(func.pg_advisory_xact_lock(func.hashtext(f"report:{key}"))))
    report = s.scalar(select(Report).where(
        Report.company_id == company.id, Report.fiscal_year == body.fiscal_year,
        Report.fiscal_period == body.fiscal_period,
        Report.status.in_(("queued", "running") if body.rerun else ("queued", "running", "done")),
    ).order_by(Report.created_at.desc()).limit(1))
    if report is None:
        queued = s.scalar(select(func.count()).where(Report.status == "queued"))
        if queued >= MAX_QUEUED:
            raise HTTPException(429, f"{queued} reports already queued; retry later",
                                headers={"Retry-After": "60"})
        report = Report(company_id=company.id, fiscal_year=body.fiscal_year,
                        fiscal_period=body.fiscal_period, status="queued")
        s.add(report)
    s.commit()
    response.headers["Location"] = f"/v1/reports/{report.id}"
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
    if cursor is not None:
        ts, id_ = _decode(cursor)
        try:
            key = uuid.UUID(id_)
        except ValueError:
            raise _invalid(("query", "cursor"), "invalid cursor", cursor) from None
        q = q.where(tuple_(Report.created_at, Report.id) < tuple_(ts, key))
    rows = s.execute(q.order_by(Report.created_at.desc(), Report.id.desc()).limit(limit + 1)).all()
    fields = ReportSummary.model_fields.keys() - {"ticker"}
    items = [ReportSummary(ticker=t, **{f: getattr(r, f) for f in fields}) for r, t in rows[:limit]]
    last = rows[limit - 1][0] if len(rows) > limit else None
    return Page(items=items, next_cursor=_encode(last.created_at, last.id) if last else None)


@app.get("/v1/claims/{claim_id}")
def get_claim(claim_id: ClaimId, s: DB) -> ClaimDetail:
    """The verdict, evidence and trace come from the latest `done` report that verified the
    claim: a queued, running or failed report's partial output is never shown here."""
    claim = s.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(404, "claim not found")
    verdict = s.scalar(select(Verdict).join(Report).where(
        Verdict.claim_id == claim_id, Report.status == "done",
    ).order_by(Verdict.id.desc()).limit(1))
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
    if cursor is not None:
        ts, id_ = _decode(cursor)
        # ASCII and length-bounded by _decode, so int() cannot blow up; the range check keeps
        # it inside Postgres integer (else a DataError, a 500).
        if not (id_.isdigit() and 1 <= int(id_) <= INT4_MAX):
            raise _invalid(("query", "cursor"), "invalid cursor", cursor)
        q = q.where(tuple_(Claim.created_at, Claim.id) < tuple_(ts, int(id_)))
    rows = s.scalars(q.order_by(Claim.created_at.desc(), Claim.id.desc()).limit(limit + 1)).all()
    last = rows[limit - 1] if len(rows) > limit else None
    return Page(items=[ClaimOut.model_validate(c) for c in rows[:limit]],
                next_cursor=_encode(last.created_at, last.id) if last else None)


@app.get("/v1/evals/latest")
def evals_latest(s: DB) -> list[EvalRunOut]:
    """Public on purpose: the eval numbers are the product's credibility (§11, §20).
    The latest run of each eval: L1 alone has four (retrieval, ner, wer, forecast)."""
    runs = s.scalars(
        select(EvalRun).distinct(EvalRun.layer, EvalRun.name)
        .order_by(EvalRun.layer, EvalRun.name, EvalRun.created_at.desc(), EvalRun.id.desc())
    ).all()
    return [EvalRunOut.model_validate(r) for r in runs]

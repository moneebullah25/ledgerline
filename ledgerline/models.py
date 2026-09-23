"""The §9 schema. One Postgres holds relational facts, full-text, and vectors (ADR-0001).

Provenance columns (source_url, fetched_at, sha256, license_note) are NOT NULL on every
ingested row (FR-6). Idempotency keys are unique constraints, so a re-run cannot insert
a duplicate even if application code forgets to check (FR-7).
"""

import uuid
from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBED_DIM = 384  # bge-small-en-v1.5

FISCAL_PERIODS = ("Q1", "Q2", "Q3", "Q4", "FY")
CLAIM_TYPES = ("reported_result", "comparison", "guidance", "qualitative")
VERDICT_LABELS = ("supported", "contradicted", "unverifiable")
STANCES = ("supports", "contradicts", "insufficient")
AGENTS = ("filing_rag", "xbrl_table", "forecast", "market_reaction", "critic")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class Provenance:
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sha256: Mapped[str] = mapped_column(String(64))
    license_note: Mapped[str] = mapped_column(Text)


class Company(Base):
    __tablename__ = "company"
    id: Mapped[int] = mapped_column(primary_key=True)
    cik: Mapped[str] = mapped_column(String(10), unique=True)  # zero-padded
    ticker: Mapped[str] = mapped_column(String(10), unique=True)
    name: Mapped[str] = mapped_column(Text)
    fiscal_year_end: Mapped[str | None] = mapped_column(String(4))  # MMDD


class Filing(Provenance, Base):
    """One parsed document. For 10-K/10-Q it is the primary document; for an 8-K earnings
    release (Item 2.02) it is exhibit 99.1, the management narrative that claims come from."""

    __tablename__ = "filing"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("company.id"), index=True)
    accession_no: Mapped[str] = mapped_column(String(20), unique=True)
    form_type: Mapped[str] = mapped_column(String(10))
    filed_at: Mapped[date] = mapped_column(Date)
    period_of_report: Mapped[date | None] = mapped_column(Date)
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    fiscal_period: Mapped[str | None] = mapped_column(String(2))
    doc_url: Mapped[str] = mapped_column(Text)
    storage_path: Mapped[str] = mapped_column(Text)  # content-addressed: objects/<sha256>
    __table_args__ = (
        CheckConstraint("form_type IN ('10-K','10-Q','8-K','10-K/A','10-Q/A')", name="form"),
    )


class XbrlFact(Provenance, Base):
    """One reported value from SEC companyfacts.

    fiscal_year/fiscal_period describe the period the VALUE covers, not the filing it came
    from: a 10-Q also restates the prior-year quarter under the same fy/fp. duration_days
    separates a 3-month quarter from a 6/9-month year-to-date figure (NULL for instants).
    derived=True marks Q4 values computed as FY minus Q1..Q3 (companies do not file a Q4 10-Q).
    """

    __tablename__ = "xbrl_fact"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("company.id"))
    tag: Mapped[str] = mapped_column(Text)  # as filed, e.g. us-gaap:Revenues
    unit: Mapped[str] = mapped_column(String(20))
    value: Mapped[Decimal] = mapped_column(Numeric(28, 6))
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    duration_days: Mapped[int | None] = mapped_column(Integer)
    fiscal_year: Mapped[int] = mapped_column(Integer)
    fiscal_period: Mapped[str] = mapped_column(String(2))
    form: Mapped[str] = mapped_column(String(10))
    accession_no: Mapped[str] = mapped_column(String(20))
    filed: Mapped[date] = mapped_column(Date)
    derived: Mapped[bool] = mapped_column(default=False)
    __table_args__ = (
        UniqueConstraint(
            "company_id", "tag", "unit", "period_start", "period_end", "accession_no",
            name="uq_xbrl_fact", postgresql_nulls_not_distinct=True,
        ),
        Index("ix_xbrl_lookup", "company_id", "tag", "unit", "fiscal_year", "fiscal_period"),
        CheckConstraint(_in("fiscal_period", FISCAL_PERIODS), name="fp"),
    )


class Chunk(Base):
    __tablename__ = "chunk"
    id: Mapped[int] = mapped_column(primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filing.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[str] = mapped_column(Text)  # e.g. "Part I > Item 2. MD&A"
    char_start: Mapped[int] = mapped_column(Integer)  # offsets into the stored document text
    char_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    __table_args__ = (
        UniqueConstraint("filing_id", "chunk_index", name="uq_chunk"),
        Index(
            "ix_chunk_embedding", "embedding", postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunk_tsv", "tsv", postgresql_using="gin"),
    )


class CallAudio(Provenance, Base):
    __tablename__ = "call_audio"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("company.id"))
    kind: Mapped[str] = mapped_column(String(20))  # earnings_call | fomc
    event_date: Mapped[date | None] = mapped_column(Date)
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    fiscal_period: Mapped[str | None] = mapped_column(String(2))
    storage_path: Mapped[str] = mapped_column(Text)
    duration_s: Mapped[float | None] = mapped_column(Float)
    speech_s: Mapped[float | None] = mapped_column(Float)  # after VAD
    reference_transcript_path: Mapped[str | None] = mapped_column(Text)  # FOMC ground truth
    __table_args__ = (UniqueConstraint("sha256", name="uq_call_audio_sha"),)


class Segment(Base):
    """A contiguous speaker turn. words holds [{"w": str, "s": float, "e": float}] so every
    token resolves back to (segment_id, t_start, t_end) (FR-14)."""

    __tablename__ = "segment"
    id: Mapped[int] = mapped_column(primary_key=True)
    call_audio_id: Mapped[int] = mapped_column(ForeignKey("call_audio.id", ondelete="CASCADE"))
    idx: Mapped[int] = mapped_column(Integer)
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    speaker_label: Mapped[str] = mapped_column(String(20))
    role: Mapped[str] = mapped_column(String(20), default="unknown")
    section: Mapped[str] = mapped_column(String(20), default="prepared_remarks")
    text: Mapped[str] = mapped_column(Text)
    words: Mapped[list[dict]] = mapped_column(JSONB)
    __table_args__ = (
        UniqueConstraint("call_audio_id", "idx", name="uq_segment"),
        CheckConstraint("role IN ('operator','executive','analyst','unknown')", name="role"),
        CheckConstraint("section IN ('prepared_remarks','qa')", name="section"),
    )


class Claim(Base):
    __tablename__ = "claim"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("company.id"), index=True)
    # The quarter whose call/release made the claim.
    fiscal_year: Mapped[int] = mapped_column(Integer)
    fiscal_period: Mapped[str] = mapped_column(String(2))
    filing_id: Mapped[int | None] = mapped_column(ForeignKey("filing.id"))
    segment_id: Mapped[int | None] = mapped_column(ForeignKey("segment.id"))
    span_start: Mapped[int] = mapped_column(Integer)  # char offsets in the source text
    span_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(20))
    metric: Mapped[str | None] = mapped_column(String(40))  # canonical concept
    value: Mapped[Decimal | None] = mapped_column(Numeric(28, 6))
    value_high: Mapped[Decimal | None] = mapped_column(Numeric(28, 6))  # ranges ("$94-98B")
    unit: Mapped[str | None] = mapped_column(String(20))
    # The period the claim is ABOUT (differs from fiscal_* for guidance and comparisons).
    target_fiscal_year: Mapped[int | None] = mapped_column(Integer)
    target_fiscal_period: Mapped[str | None] = mapped_column(String(2))
    comparison: Mapped[str | None] = mapped_column(String(10))  # yoy | qoq | None
    hedging_score: Mapped[float] = mapped_column(Float, default=0.0)
    speaker_role: Mapped[str | None] = mapped_column(String(20))
    linked_claim_id: Mapped[int | None] = mapped_column(ForeignKey("claim.id"))
    extractor_version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        CheckConstraint(_in("claim_type", CLAIM_TYPES), name="claim_type"),
        CheckConstraint("(filing_id IS NULL) <> (segment_id IS NULL)", name="one_source"),
        UniqueConstraint(
            "filing_id", "segment_id", "span_start", "span_end", "extractor_version",
            name="uq_claim_span", postgresql_nulls_not_distinct=True,
        ),
    )


class Report(Base):
    """One run of the pipeline for (company, quarter). id doubles as the job_id (§15)."""

    __tablename__ = "report"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id: Mapped[int] = mapped_column(ForeignKey("company.id"))
    fiscal_year: Mapped[int] = mapped_column(Integer)
    fiscal_period: Mapped[str] = mapped_column(String(2))
    status: Mapped[str] = mapped_column(String(10), default="queued")
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_s: Mapped[float | None] = mapped_column(Float)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','done','failed')", name="status"),
        Index("ix_report_created", "created_at", "id"),
    )


class Evidence(Base):
    __tablename__ = "evidence"
    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("report.id", ondelete="CASCADE"))
    claim_id: Mapped[int] = mapped_column(ForeignKey("claim.id"), index=True)
    agent: Mapped[str] = mapped_column(String(20))
    stance: Mapped[str] = mapped_column(String(15))
    summary: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    model: Mapped[str | None] = mapped_column(String(40))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    trace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    __table_args__ = (
        CheckConstraint(_in("agent", AGENTS), name="agent"),
        CheckConstraint(_in("stance", STANCES), name="stance"),
    )


class Verdict(Base):
    __tablename__ = "verdict"
    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("report.id", ondelete="CASCADE"))
    claim_id: Mapped[int] = mapped_column(ForeignKey("claim.id"))
    label: Mapped[str] = mapped_column(String(15))
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict]] = mapped_column(JSONB)
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    refused: Mapped[bool] = mapped_column(default=False)  # FR-44 short-circuit, no adjudicator
    prompt_version: Mapped[str] = mapped_column(String(40))
    model: Mapped[str | None] = mapped_column(String(40))
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    trace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    __table_args__ = (
        UniqueConstraint("report_id", "claim_id", name="uq_verdict"),
        CheckConstraint(_in("label", VERDICT_LABELS), name="label"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence"),
    )


class Span(Base):
    """Trace span. Every graph node, LLM call included, writes one (§12)."""

    __tablename__ = "span"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    name: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    attrs: Mapped[dict] = mapped_column(JSONB, default=dict)


class MarketPrice(Base):
    __tablename__ = "market_price"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10))
    day: Mapped[date] = mapped_column(Date)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(Numeric(20, 0))
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sha256: Mapped[str] = mapped_column(String(64))  # of the chart response the row came from
    license_note: Mapped[str] = mapped_column(Text)
    __table_args__ = (UniqueConstraint("ticker", "day", name="uq_price"),)


class IngestFailure(Base):
    """A permanent ingest failure recorded as data instead of a crash (FR-7, §8)."""

    __tablename__ = "ingest_failure"
    id: Mapped[int] = mapped_column(primary_key=True)
    flow: Mapped[str] = mapped_column(String(40))
    key: Mapped[str] = mapped_column(Text)
    error_class: Mapped[str] = mapped_column(String(60))  # http_4xx | schema_drift | parse | ...
    error: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("flow", "key", "error_class", name="uq_ingest_failure"),)


class EvalRun(Base):
    __tablename__ = "eval_run"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    git_sha: Mapped[str | None] = mapped_column(String(40))
    layer: Mapped[str] = mapped_column(String(4))  # L1..L4
    # Which eval of the layer (retrieval, ner, wer, forecast share L1). Defaults to the layer
    # so writers that do not set it keep working.
    name: Mapped[str] = mapped_column(
        String(40), default=lambda ctx: ctx.get_current_parameters()["layer"])
    metrics: Mapped[dict] = mapped_column(JSONB)
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    __table_args__ = (Index("ix_eval_run_latest", "layer", "name", "created_at"),)

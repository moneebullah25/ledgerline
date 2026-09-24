"""The capstone flow (§4): one company-quarter in, one ranked, cited report out.

    python -m ledgerline.pipeline --ticker AAPL --fy 2025 --fp Q3

Transactions: one commit per claim. A claim's evidence, verdict and trace spans land
together or not at all, and a crash keeps every claim finished before it. A re-run of the
same report skips claims that already have a verdict, so the worker can safely re-queue a
report whose process died (see worker.py). Each extracted filing/call and each claim commit
also refreshes report.config["heartbeat_at"], which is what the worker's stale sweep watches.
report.config["mode"] records whether the run had LLM credentials (online/offline).

A claim whose verification raises (citation-integrity failure (FR-45), model refusal, API
or overload error, adjudicator wiring bug) is stored as a refused `unverifiable` verdict
naming the error class, and the report goes on. Only database errors, or a missing
dependency (ImportError), fail the report: they would fail every claim the same way.
"""

import argparse
import logging
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy import exists, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ledgerline import llm
from ledgerline.llm import Usage
from ledgerline.models import CallAudio, Claim, Company, Evidence, Filing, Report, Segment, Verdict
from ledgerline.report.generate import NO_VERDICT
from ledgerline.report.severity import load_weights, severity
from ledgerline.tracing import Tracer, error_summary

log = logging.getLogger("ledgerline.pipeline")
FY_UNSUPPORTED = ("FY reports are not supported: the full year is covered by the fourth-quarter "
                  "earnings release, so request Q4")


def _ensure_claims(s: Session, report: Report) -> list[Claim]:
    """Extract claims for the quarter's earnings release(s) and call(s) that have none at
    the current extractor version, then return every claim of that version for the quarter."""
    from ledgerline.extract.pipeline import EXTRACTOR_VERSION, extract_for_call, extract_for_filing

    current = (Claim.extractor_version == EXTRACTOR_VERSION,)
    releases = s.scalars(select(Filing.id).where(
        Filing.company_id == report.company_id, Filing.form_type == "8-K",
        Filing.fiscal_year == report.fiscal_year, Filing.fiscal_period == report.fiscal_period,
        ~exists().where(Claim.filing_id == Filing.id, *current),
    )).all()
    for filing_id in releases:
        extract_for_filing(s, filing_id)
        _heartbeat(s, report)
        s.commit()

    calls = s.scalars(select(CallAudio.id).where(
        CallAudio.company_id == report.company_id, CallAudio.kind == "earnings_call",
        CallAudio.fiscal_year == report.fiscal_year,
        CallAudio.fiscal_period == report.fiscal_period,
        ~exists().where(Claim.segment_id == Segment.id,
                        Segment.call_audio_id == CallAudio.id, *current),
    )).all()
    for call_id in calls:
        if not s.scalar(select(exists().where(Segment.call_audio_id == call_id))):
            from ledgerline.audio import process_call  # needs the [ml] extra

            process_call(s, call_id)
        extract_for_call(s, call_id)
        _heartbeat(s, report)
        s.commit()

    return list(s.scalars(select(Claim).where(
        Claim.company_id == report.company_id, Claim.fiscal_year == report.fiscal_year,
        Claim.fiscal_period == report.fiscal_period, *current,
    ).order_by(Claim.id)))


def _deviation(result) -> float | None:
    """The XBRL agent's payload["deviation"] (signed; severity wants the magnitude)."""
    for e in result.evidence:
        if (e.agent == "xbrl_table" and e.stance != "insufficient"
                and (d := e.payload.get("deviation")) is not None):
            return abs(float(d))
    return None


def _abnormal_return(result) -> float | None:
    for e in result.evidence:
        if e.agent == "market_reaction" and (ar := e.payload.get("abnormal_return")) is not None:
            return float(ar)
    return None


def _persist(s: Session, report: Report, claim: Claim, result, tracer: Tracer) -> None:
    usage: dict[str, list[Usage]] = defaultdict(list)
    for name, u in result.usages:
        usage[name].append(u)
    # Agents trace as `agent.<name>`; the critic's span is plain `critic`.
    latency = {sp.name: sp.attrs.get("latency_ms", 0) for sp in tracer.spans}

    def row(agent: str, stance: str, summary: str, citations: list, payload: dict) -> Evidence:
        us = usage.get(agent, [])
        return Evidence(
            report_id=report.id, claim_id=claim.id, agent=agent, stance=stance,
            summary=summary, citations=citations, payload=payload,
            model=next((u.model for u in us if u.model), None),
            input_tokens=sum(u.input_tokens for u in us),
            output_tokens=sum(u.output_tokens for u in us),
            cache_read_tokens=sum(u.cache_read_tokens for u in us),
            cost_usd=sum(u.cost_usd for u in us),
            latency_ms=latency.get(f"agent.{agent}", latency.get(agent, 0)),
            trace_id=tracer.trace_id,
        )

    for e in result.evidence:
        s.add(row(e.agent, e.stance, e.summary,
                  [c.model_dump() for c in e.citations], e.payload))
    if result.critic is not None:
        c = result.critic
        s.add(row("critic", c.counter_stance, c.argument, [],
                  {"weakest_evidence": c.weakest_evidence}))

    v = result.verdict
    s.add(Verdict(
        report_id=report.id, claim_id=claim.id, label=v.label, confidence=v.confidence,
        rationale=v.rationale, citations=[c.model_dump() for c in v.citations],
        severity=severity(v.label, v.confidence, _deviation(result), claim.hedging_score,
                          _abnormal_return(result), load_weights()),
        refused=result.refused, prompt_version=result.prompt_version, model=result.model,
        cost_usd=result.cost_usd, trace_id=tracer.trace_id,
    ))
    tracer.flush(s)


def _persist_no_verdict(s: Session, report: Report, claim: Claim, tracer: Tracer,
                        prompt_version: str, err: Exception, reason: str) -> None:
    """FR-45 held: no verdict text that failed the audit is stored or shown. The spend is:
    what the finished nodes recorded on their spans, plus the failed call's usage if the
    error carries one (the verify_claim span only gets a total on success)."""
    usage = getattr(err, "usage", None)
    cost = sum(sp.attrs.get("cost_usd", 0.0) for sp in tracer.spans if sp.name != "verify_claim")
    cost += usage.cost_usd if usage is not None else 0.0
    s.add(Verdict(
        report_id=report.id, claim_id=claim.id, label="unverifiable", confidence=0.0,
        rationale=f"{NO_VERDICT}{reason}.",
        citations=[], severity=0.0, refused=True, prompt_version=prompt_version, model=None,
        cost_usd=cost, trace_id=tracer.trace_id,
    ))
    tracer.flush(s)


def _heartbeat(s: Session, report: Report, **extra) -> None:
    """Re-read config first: the worker writes `attempts` there, and a stale in-memory
    dict written back would clobber it."""
    s.refresh(report, ["config"])
    report.config = {**report.config, **extra, "heartbeat_at": datetime.now(UTC).isoformat()}


def _total_cost(s: Session, report_id: uuid.UUID) -> float:
    # Verdict.cost_usd is the claim's whole spend (agents, critic, adjudicator), so it is
    # recomputed from committed rows rather than accumulated in memory a crash would lose.
    return s.scalar(select(func.coalesce(func.sum(Verdict.cost_usd), 0.0))
                    .where(Verdict.report_id == report_id))


def run_report(s: Session, report_id: uuid.UUID) -> None:
    """Runs to completion or marks the report failed; never leaves it 'running' unless the
    process itself dies (the worker's stale sweep covers that)."""
    report = s.get(Report, report_id)
    if report is None:
        raise ValueError(f"no report {report_id}")
    t0 = time.perf_counter()
    report.status = "running"
    _heartbeat(s, report, started_at=datetime.now(UTC).isoformat(),
               mode="online" if llm.available() else "offline")
    s.commit()
    try:
        # Inside try: a broken import fails the report.
        from ledgerline.graph import load_prompt
        from ledgerline.graph.adjudicator import IntegrityError as CitationIntegrityError
        from ledgerline.graph.build import verify_claim

        if report.fiscal_period == "FY":
            raise ValueError(FY_UNSUPPORTED)
        claims = _ensure_claims(s, report)
        if not claims:
            raise LookupError(
                f"no claims for FY{report.fiscal_year} {report.fiscal_period}: no 8-K earnings "
                "release or call audio ingested for that quarter (run make backfill)")
        done = set(s.scalars(select(Verdict.claim_id).where(Verdict.report_id == report_id)))
        for claim in claims:
            if claim.id in done:
                continue
            tracer = Tracer()
            try:
                result = verify_claim(s, claim, tracer)
            except (SQLAlchemyError, ImportError):
                raise
            except Exception as e:
                log.exception("report %s claim %s: no verdict", report_id, claim.id)
                reason = ("the generated rationale failed the citation-integrity check"
                          if isinstance(e, CitationIntegrityError)
                          else f"{type(e).__name__} while verifying the claim")
                _persist_no_verdict(s, report, claim, tracer, load_prompt("adjudicator")[1],
                                    e, reason)
            else:
                _persist(s, report, claim, result, tracer)
            _heartbeat(s, report)
            s.commit()
        report.status, report.error = "done", None
    except Exception as e:  # any failure must end as status='failed', not 'running'
        log.exception("report %s failed", report_id)  # the traceback stays server-side
        s.rollback()  # drops only the claim in flight; earlier claims are committed
        report.status, report.error = "failed", error_summary(e)
    report.cost_usd = _total_cost(s, report_id)
    report.latency_s = round(time.perf_counter() - t0, 3)
    report.finished_at = datetime.now(UTC)
    s.commit()


def main(argv: list[str] | None = None) -> int:
    from ledgerline.db import engine
    from ledgerline.report.generate import build_report, to_markdown

    p = argparse.ArgumentParser(prog="python -m ledgerline.pipeline", description=__doc__)
    p.add_argument("--ticker", required=True)
    p.add_argument("--fy", type=int, required=True)
    p.add_argument("--fp", required=True, choices=["Q1", "Q2", "Q3", "Q4", "FY"])
    a = p.parse_args(argv)
    if a.fp == "FY":
        p.error(FY_UNSUPPORTED)
    with Session(engine(), expire_on_commit=False) as s:
        company = s.scalar(select(Company).where(Company.ticker == a.ticker.upper()))
        if company is None:
            print(f"unknown ticker {a.ticker}: run `make backfill TICKER={a.ticker}` first",
                  file=sys.stderr)
            return 2
        report = Report(company_id=company.id, fiscal_year=a.fy, fiscal_period=a.fp)
        s.add(report)
        s.commit()
        run_report(s, report.id)
        view = build_report(s, report.id)
    print(to_markdown(view))
    return 0 if view.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())

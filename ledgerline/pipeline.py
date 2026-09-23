"""The capstone flow (§4): one company-quarter in, one ranked, cited report out.

    python -m ledgerline.pipeline --ticker AAPL --fy 2025 --fp Q3

Transactions: one commit per claim. A claim's evidence, verdict and trace spans land
together or not at all, and a crash keeps every claim finished before it. A re-run of the
same report skips claims that already have a verdict, so the worker can safely re-queue a
report whose process died (see worker.py).
"""

import argparse
import sys
import time
import traceback
import uuid
from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from ledgerline.llm import Usage
from ledgerline.models import CallAudio, Claim, Company, Evidence, Filing, Report, Segment, Verdict
from ledgerline.report.severity import load_weights, severity
from ledgerline.tracing import Tracer


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

    return list(s.scalars(select(Claim).where(
        Claim.company_id == report.company_id, Claim.fiscal_year == report.fiscal_year,
        Claim.fiscal_period == report.fiscal_period, *current,
    ).order_by(Claim.id)))


def _deviation(claim: Claim, result) -> float | None:
    """Relative error from the XBRL agent: its payload value if it computed one, else from
    the claim's number and the first fact value it cited."""
    for e in result.evidence:
        if e.agent != "xbrl_table" or e.stance == "insufficient":
            continue
        if (rel := e.payload.get("relative_error")) is not None:
            return float(rel)
        if claim.value is not None:
            for c in e.citations:
                # Same unit only: a margin claim may cite its USD component facts.
                if c.kind == "xbrl_fact" and c.value and c.unit in (None, claim.unit):
                    return abs(float(claim.value) - c.value) / abs(c.value)
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
            cost_usd=sum(u.cost_usd for u in us), latency_ms=latency.get(agent, 0),
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
        severity=severity(v.label, v.confidence, _deviation(claim, result), claim.hedging_score,
                          _abnormal_return(result), load_weights()),
        refused=result.refused, prompt_version=result.prompt_version, model=result.model,
        cost_usd=result.cost_usd, trace_id=tracer.trace_id,
    ))
    tracer.flush(s)


def run_report(s: Session, report_id: uuid.UUID) -> None:
    """Runs to completion or marks the report failed; never leaves it 'running' unless the
    process itself dies (the worker's stale sweep covers that)."""
    report = s.get(Report, report_id)
    if report is None:
        raise ValueError(f"no report {report_id}")
    t0 = time.perf_counter()
    report.status = "running"
    report.config = {**report.config, "started_at": datetime.now(UTC).isoformat()}
    s.commit()
    cost = 0.0
    try:
        from ledgerline.graph.build import (
            verify_claim,  # inside try: a broken import fails the report
        )

        claims = _ensure_claims(s, report)
        s.commit()
        if not claims:
            raise LookupError(
                f"no claims for FY{report.fiscal_year} {report.fiscal_period}: no 8-K earnings "
                "release or call audio ingested for that quarter (run make backfill)")
        done = set(s.scalars(select(Verdict.claim_id).where(Verdict.report_id == report_id)))
        for claim in claims:
            if claim.id in done:
                continue
            tracer = Tracer()
            with tracer.span("verify_claim", claim_id=claim.id, report_id=str(report_id)):
                result = verify_claim(s, claim, tracer)
            _persist(s, report, claim, result, tracer)
            s.commit()
            cost += result.cost_usd
        report.status, report.error = "done", None
    except Exception:  # noqa: BLE001 - any failure must end as status='failed', not 'running'
        s.rollback()  # drops only the claim in flight; earlier claims are committed
        report.status, report.error = "failed", traceback.format_exc()[-4000:]  # tail keeps the error
    # Cost of this attempt; a resumed report adds to what the earlier attempt spent.
    report.cost_usd = (report.cost_usd or 0.0) + cost
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

"""The capstone flow (§4): one company-quarter in, one ranked, cited report out.

    python -m ledgerline.pipeline --ticker AAPL --fy 2025 --fp Q3

Transactions: one commit per claim. A claim's evidence, verdict and trace spans land
together or not at all, and a crash keeps every claim finished before it. A re-run of the
same report skips claims that already have a verdict, so the worker can safely re-queue a
report whose process died (see worker.py). Each extracted filing/call and each claim commit
also refreshes report.config["heartbeat_at"], which is what the worker's stale sweep watches;
so do the extraction's progress callbacks and a timer while claims are in flight.
report.config["mode"] records whether the run had LLM credentials (online/offline).

Claims are verified in parallel (db.VERIFY_WORKERS threads), each on its own Session; the
evidence, verdict and heartbeat are written and committed on the calling thread as claims
finish. A caller that hands in a Session bound to one Connection (tests) gets one worker.

Budget guard (NFR-3): once verification spend runs ahead of the budget's pace (spend >
(budget - extraction spend) x finished/total claims), the claims submitted after that use the
offline critic and adjudicator; report.config["budget_guard"] records when it fired.

A claim whose verification raises (citation-integrity failure (FR-45), model refusal,
adjudicator wiring bug) is stored as a refused `unverifiable` verdict naming the error class,
and the report goes on. Infrastructure errors fail the report instead, since they would fail
every claim the same way: database errors, a missing dependency (ImportError), and an
unreachable, rate-limited or unauthenticated model (llm.LLMUnavailable). A failed report is
re-run by POSTing it again.
"""

import argparse
import logging
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime

from sqlalchemy import Engine, exists, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ledgerline import llm
from ledgerline.config import settings
from ledgerline.db import VERIFY_WORKERS
from ledgerline.llm import Usage
from ledgerline.logging import configure_logging
from ledgerline.models import CallAudio, Claim, Company, Evidence, Filing, Report, Segment, Verdict
from ledgerline.report.generate import COST_BUDGET_USD, NO_VERDICT
from ledgerline.report.severity import load_weights, severity
from ledgerline.tracing import Tracer, error_summary

log = logging.getLogger("ledgerline.pipeline")
FY_UNSUPPORTED = ("FY reports are not supported: the full year is covered by the fourth-quarter "
                  "earnings release, so request Q4")
INFRA_ERRORS = (SQLAlchemyError, ImportError, llm.LLMUnavailable)
HEARTBEAT_S = 60  # refresh the heartbeat at least this often while claims are in flight


def _ensure_claims(s: Session, report: Report) -> list[Claim]:
    """Extract claims for the quarter's earnings release(s) and call(s) that have none at
    the current extractor version, then return every claim of that version for the quarter.
    The extraction's LLM spend accumulates in report.config["extract_cost_usd"], committed
    with each source's claims (a crash loses both; the rerun's refine answers are cached)."""
    from ledgerline.extract.pipeline import extract_for_call, extract_for_filing, extractor_version

    current = (Claim.extractor_version == extractor_version(),)
    spent: list[float] = []

    def progress(batch: Usage) -> None:  # per batch of LLM refine answers
        spent.append(batch.cost_usd)
        _touch(s, report.id)

    def done_with_source() -> None:
        _heartbeat(s, report, extract_cost_usd=report.config.get("extract_cost_usd", 0.0)
                   + sum(spent))
        spent.clear()
        s.commit()

    releases = s.scalars(select(Filing.id).where(
        Filing.company_id == report.company_id, Filing.form_type == "8-K",
        Filing.fiscal_year == report.fiscal_year, Filing.fiscal_period == report.fiscal_period,
        ~exists().where(Claim.filing_id == Filing.id, *current),
    )).all()
    for filing_id in releases:
        extract_for_filing(s, filing_id, on_progress=progress)
        done_with_source()

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
        extract_for_call(s, call_id, on_progress=progress)
        done_with_source()

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


def _persist(s: Session, report: Report, claim: Claim, result, tracer: Tracer) -> Verdict:
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
            cache_write_tokens=sum(u.cache_write_tokens for u in us),
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
    row = Verdict(
        report_id=report.id, claim_id=claim.id, label=v.label, confidence=v.confidence,
        rationale=v.rationale, citations=[c.model_dump() for c in v.citations],
        severity=severity(v.label, v.confidence, _deviation(result), claim.hedging_score,
                          _abnormal_return(result), load_weights()),
        refused=result.refused, prompt_version=result.prompt_version, model=result.model,
        cost_usd=result.cost_usd, trace_id=tracer.trace_id,
    )
    s.add(row)
    tracer.flush(s)
    return row


def _persist_no_verdict(s: Session, report: Report, claim: Claim, tracer: Tracer,
                        prompt_version: str, err: Exception, reason: str) -> Verdict:
    """FR-45 held: no verdict text that failed the audit is stored or shown. The spend is:
    what the finished nodes recorded on their spans, plus the failed call's usage if the
    error carries one (the verify_claim span only gets a total on success)."""
    usage = getattr(err, "usage", None)
    cost = sum(sp.attrs.get("cost_usd", 0.0) for sp in tracer.spans if sp.name != "verify_claim")
    cost += usage.cost_usd if usage is not None else 0.0
    row = Verdict(
        report_id=report.id, claim_id=claim.id, label="unverifiable", confidence=0.0,
        rationale=f"{NO_VERDICT}{reason}.",
        citations=[], severity=0.0, refused=True, prompt_version=prompt_version, model=None,
        cost_usd=cost, trace_id=tracer.trace_id,
    )
    s.add(row)
    tracer.flush(s)
    return row


def _touch(s: Session, report_id: uuid.UUID) -> None:
    """Heartbeat from inside a long step whose own transaction must not commit yet
    (extraction): a separate short transaction, merging into config so `attempts` survives.
    A Session bound to one Connection (tests) has no second connection to use: skipped."""
    bind = s.get_bind()
    if not isinstance(bind, Engine):
        return
    beat = func.jsonb_build_object("heartbeat_at", datetime.now(UTC).isoformat())
    with bind.begin() as c:
        c.execute(update(Report).where(Report.id == report_id)
                  .values(config=Report.config.op("||")(beat)))


def _heartbeat(s: Session, report: Report, **extra) -> None:
    """Re-read config first: the worker writes `attempts` there, and a stale in-memory
    dict written back would clobber it."""
    s.refresh(report, ["config"])
    report.config = {**report.config, **extra, "heartbeat_at": datetime.now(UTC).isoformat()}


def _verdict_cost(s: Session, report_id: uuid.UUID) -> float:
    # Verdict.cost_usd is the claim's whole spend (agents, critic, adjudicator), so it is
    # recomputed from committed rows rather than accumulated in memory a crash would lose.
    return s.scalar(select(func.coalesce(func.sum(Verdict.cost_usd), 0.0))
                    .where(Verdict.report_id == report_id))


def _verify(verify_claim, bind, claim_id: int, tracer: Tracer, offline: bool):
    """One claim on a worker thread, on its own Session."""
    # create_savepoint: on a caller's Connection (tests) join its transaction, never end it.
    with Session(bind, expire_on_commit=False, join_transaction_mode="create_savepoint") as ws:
        return verify_claim(ws, ws.get(Claim, claim_id), tracer, offline=offline)


def _verify_claims(s: Session, report: Report, claims: list[Claim]) -> None:
    """Verifies every claim without a verdict yet (up to VERIFY_WORKERS at a time) and
    writes each on this thread as it finishes. Raises only on infrastructure errors."""
    from ledgerline.graph import load_prompt
    from ledgerline.graph.adjudicator import IntegrityError as CitationIntegrityError
    from ledgerline.graph.build import verify_claim

    done = set(s.scalars(select(Verdict.claim_id).where(Verdict.report_id == report.id)))
    todo = [c for c in claims if c.id not in done]
    extract_cost = report.config.get("extract_cost_usd", 0.0)
    spent, finished = _verdict_cost(s, report.id), len(done)
    guard = report.config.get("budget_guard")  # sticky, also across a resumed run
    bind = s.get_bind()
    # A Session on one Connection cannot be shared across threads: one claim at a time, and
    # never submitted while this thread writes.
    workers = VERIFY_WORKERS if isinstance(bind, Engine) else 1
    running: dict = {}
    with ThreadPoolExecutor(workers) as pool:
        while todo or running:
            while todo and len(running) < workers:
                claim, tracer = todo.pop(0), Tracer()  # read here after the worker is done
                running[pool.submit(_verify, verify_claim, bind, claim.id, tracer,
                                    guard is not None)] = (claim, tracer)
            ready, _ = wait(running, timeout=HEARTBEAT_S, return_when=FIRST_COMPLETED)
            for future in ready:
                claim, tracer = running.pop(future)
                try:
                    result = future.result()
                except INFRA_ERRORS:
                    raise  # leaving the with-block waits for the claims still in flight
                except Exception as e:
                    log.exception("report %s claim %s: no verdict", report.id, claim.id)
                    reason = ("the generated rationale failed the citation-integrity check"
                              if isinstance(e, CitationIntegrityError)
                              else f"{type(e).__name__} while verifying the claim")
                    v = _persist_no_verdict(s, report, claim, tracer,
                                            load_prompt("adjudicator")[1], e, reason)
                else:
                    v = _persist(s, report, claim, result, tracer)
                spent += v.cost_usd
                finished += 1
                pace = (COST_BUDGET_USD - extract_cost) * finished / len(claims)
                if guard is None and spent > pace:
                    guard = {"after_claims": finished, "of": len(claims),
                             "spent_usd": round(extract_cost + spent, 4)}
                    log.warning("report %s: budget guard fired %s", report.id, guard)
                _heartbeat(s, report, **({"budget_guard": guard} if guard else {}))
                s.commit()
            if not ready:  # nothing finished within HEARTBEAT_S, but the run is alive
                _heartbeat(s, report)
                s.commit()


def run_report(s: Session, report_id: uuid.UUID) -> None:
    """Runs to completion or marks the report failed; never leaves it 'running' unless the
    process itself dies (the worker's stale sweep covers that)."""
    configure_logging(settings())  # idempotent; no-op if already configured
    report = s.get(Report, report_id)
    if report is None:
        raise ValueError(f"no report {report_id}")
    t0 = time.perf_counter()
    report.status = "running"
    mode = "online" if llm.available() else "offline"
    log.info("report started", extra={"report_id": str(report_id), "mode": mode})
    _heartbeat(s, report, started_at=datetime.now(UTC).isoformat(), mode=mode)
    s.commit()
    try:
        if report.fiscal_period == "FY":
            raise ValueError(FY_UNSUPPORTED)
        claims = _ensure_claims(s, report)
        if not claims:
            raise LookupError(
                f"no claims for FY{report.fiscal_year} {report.fiscal_period}: no 8-K earnings "
                "release or call audio ingested for that quarter (run make backfill)")
        log.info("verifying claims", extra={"report_id": str(report_id), "claim_count": len(claims)})
        _verify_claims(s, report, claims)  # imports the graph here: a broken one fails the report
        report.status, report.error = "done", None
        log.info("report done", extra={"report_id": str(report_id)})
    except Exception as e:  # any failure must end as status='failed', not 'running'
        log.exception("report %s failed", report_id)  # the traceback stays server-side
        s.rollback()  # drops only the claim in flight; earlier claims are committed
        report.status, report.error = "failed", error_summary(e)
    report.cost_usd = _verdict_cost(s, report_id) + report.config.get("extract_cost_usd", 0.0)
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

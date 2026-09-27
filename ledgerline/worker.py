"""Report worker: `python -m ledgerline.worker`.

The queue is the report table itself (ADR-0001: one Postgres). A worker claims the oldest
queued report with SELECT ... FOR UPDATE SKIP LOCKED, flips it to 'running' and commits, so
any number of workers can poll without double-running a report.

SIGTERM/SIGINT: finish the current report, then exit. If the process is killed anyway,
the report stays 'running'; the stale sweep re-queues it once its heartbeat is older than
STALE_AFTER, and run_report resumes at the first claim without a verdict. run_report
refreshes the heartbeat after each extracted filing/call and each claim commit, so a long
but live report is never swept and run twice. After MAX_ATTEMPTS starts it is marked failed instead, so a report that
crashes the process cannot loop forever. Start time, heartbeat and attempt count live in
report.config (started_at, heartbeat_at, attempts) to avoid a schema change.
"""

import logging
import signal
import threading
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import DateTime, cast, func, select
from sqlalchemy.orm import Session

from ledgerline.db import engine
from ledgerline.models import Report

POLL_S = 2.0
STALE_AFTER = timedelta(minutes=60)  # since the last heartbeat; covers a long call-audio ASR
MAX_ATTEMPTS = 3

log = logging.getLogger("ledgerline.worker")


def claim_next(s: Session) -> uuid.UUID | None:
    report = s.scalars(
        select(Report)
        .where(Report.status == "queued")
        .order_by(Report.created_at, Report.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if report is None:
        return None
    report.status = "running"
    now = datetime.now(UTC).isoformat()
    report.config = {
        **report.config,
        "attempts": report.config.get("attempts", 0) + 1,
        "started_at": now,
        "heartbeat_at": now,
    }
    s.commit()
    return report.id


def sweep_stale(s: Session, now: datetime | None = None) -> int:
    cutoff = (now or datetime.now(UTC)) - STALE_AFTER
    stale = s.scalars(
        select(Report)
        .where(
            Report.status == "running",
            cast(
                func.coalesce(
                    Report.config["heartbeat_at"].astext, Report.config["started_at"].astext
                ),
                DateTime(timezone=True),
            )
            < cutoff,
        )
        .with_for_update(skip_locked=True)
    ).all()
    for r in stale:
        if r.config.get("attempts", 1) >= MAX_ATTEMPTS:
            r.status, r.error = "failed", f"stale: no progress after {MAX_ATTEMPTS} attempts"
            r.finished_at = datetime.now(UTC)
        else:
            r.status = "queued"
    s.commit()
    return len(stale)


def main() -> None:
    from ledgerline.pipeline import run_report

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    log.info("worker started")
    while not stop.is_set():
        with Session(engine(), expire_on_commit=False) as s:
            if n := sweep_stale(s):
                log.info("swept %d stale reports", n)
            report_id = claim_next(s)
            if report_id is None:
                stop.wait(POLL_S)
                continue
            log.info("running report %s", report_id)
            try:
                run_report(s, report_id)
                log.info("report %s: %s", report_id, s.get(Report, report_id).status)
            except Exception:
                log.exception("report %s: worker error", report_id)
                stop.wait(POLL_S)
    log.info("worker stopped")


if __name__ == "__main__":
    main()

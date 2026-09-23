from datetime import UTC, datetime, timedelta

import pytest

from ledgerline.models import Company, Report
from ledgerline.worker import MAX_ATTEMPTS, STALE_AFTER, claim_next, sweep_stale


@pytest.mark.db
def test_claim_next_takes_oldest_queued_and_sweep_recovers_stale(db):
    co = Company(cik="0000021344", ticker="KO", name="Coca-Cola")
    db.add(co)
    db.flush()
    old = datetime.now(UTC) - STALE_AFTER - timedelta(minutes=1)
    now = datetime.now(UTC)
    first = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q1",
                   created_at=now - timedelta(hours=3))
    second = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q2",
                    created_at=now - timedelta(hours=1))
    dead = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q3", status="running",
                  created_at=now - timedelta(hours=2),
                  config={"started_at": old.isoformat(), "attempts": 1})
    poison = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q4", status="running",
                    config={"started_at": old.isoformat(), "attempts": MAX_ATTEMPTS})
    db.add_all([first, second, dead, poison])
    db.flush()

    assert claim_next(db) == first.id
    assert first.status == "running" and first.config["attempts"] == 1

    assert sweep_stale(db) == 2  # `first` just started, so it is not stale
    assert dead.status == "queued"
    assert poison.status == "failed" and "stale" in poison.error
    assert claim_next(db) == dead.id  # re-queued work keeps its place: created before second
    assert dead.config["attempts"] == 2


@pytest.mark.db
def test_sweep_spares_a_long_report_with_a_fresh_heartbeat(db):
    co = Company(cik="0000021344", ticker="KO", name="Coca-Cola")
    db.add(co)
    db.flush()
    old = (datetime.now(UTC) - STALE_AFTER - timedelta(minutes=1)).isoformat()
    live = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q1", status="running",
                  config={"started_at": old, "heartbeat_at": datetime.now(UTC).isoformat(),
                          "attempts": 1})
    stuck = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q2", status="running",
                   config={"started_at": old, "heartbeat_at": old, "attempts": 1})
    db.add_all([live, stuck])
    db.flush()
    assert sweep_stale(db) == 1
    assert (live.status, stuck.status) == ("running", "queued")

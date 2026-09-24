"""run_report against fake extraction and graph modules (the real ones are built separately
and exercised by their own tests); the database side is real."""

import sys
import time
import types
from datetime import date

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from test_report_generate import PROV

from ledgerline.llm import Usage
from ledgerline.models import Claim, Company, Evidence, Filing, Report, Span, Verdict
from ledgerline.pipeline import run_report
from ledgerline.schemas import Citation, CriticOutput, EvidenceItem, VerdictDraft


class FakeResult(BaseModel):
    verdict: VerdictDraft
    evidence: list[EvidenceItem]
    critic: CriticOutput | None
    refused: bool
    prompt_version: str
    model: str | None
    cost_usd: float
    usages: list[tuple[str, Usage]]


class FakeIntegrityError(RuntimeError):
    pass


def _install_fakes(monkeypatch, *, fail_on: int | None = None,
                   integrity_fail_on: int | None = None) -> list[int]:
    extracted: list[int] = []

    def extract_for_filing(s, filing_id):
        extracted.append(filing_id)
        for i, (text, value, hedge) in enumerate([
            ("Revenue was $97.0 billion.", 97.0e9, 0.0),
            ("We expect revenue of about $100 billion.", 100e9, 0.8),
        ]):
            s.add(Claim(company_id=s.get(Filing, filing_id).company_id, fiscal_year=2025,
                        fiscal_period="Q3", filing_id=filing_id, span_start=i * 50,
                        span_end=i * 50 + 20, text=text, claim_type="reported_result",
                        metric="revenue", value=value, unit="USD", hedging_score=hedge,
                        extractor_version="fake-1"))
        s.flush()
        return 2

    calls = {"n": 0}

    def verify_claim(s, claim, tracer):
        calls["n"] += 1
        if calls["n"] == fail_on:
            raise RuntimeError("graph exploded")
        with tracer.span("verify_claim", claim_id=claim.id):  # as graph.build does
            with tracer.span("agent.xbrl_table") as a:
                a["cost_usd"] = 0.005
                time.sleep(0.002)
            if calls["n"] == integrity_fail_on:
                raise FakeIntegrityError("adjudicator output failed audit twice: [97.1]")
        fact = Citation(kind="xbrl_fact", ref_id=1, value=94.0e9, unit="USD")
        contradicted = claim.hedging_score == 0.0
        return FakeResult(
            verdict=VerdictDraft(
                label="contradicted" if contradicted else "unverifiable",
                confidence=0.9 if contradicted else 0.5, rationale="See the filing.",
                citations=[fact] if contradicted else []),
            evidence=[EvidenceItem(agent="xbrl_table", stance="contradicts",
                                   summary="lower", citations=[fact],
                                   payload={"deviation": -0.25 if contradicted else 0.0})],
            critic=CriticOutput(counter_stance="supports", argument="Rounding?"),
            refused=False, prompt_version="p1", model="claude-opus-5", cost_usd=0.02,
            usages=[("xbrl_table", Usage("claude-sonnet-5", 1000, 100, 800, 0))],
        )

    monkeypatch.setitem(sys.modules, "ledgerline.extract.pipeline", types.SimpleNamespace(
        EXTRACTOR_VERSION="fake-1", extract_for_filing=extract_for_filing,
        extract_for_call=lambda s, cid: 0))
    monkeypatch.setitem(sys.modules, "ledgerline.graph.build",
                        types.SimpleNamespace(verify_claim=verify_claim))
    monkeypatch.setattr("ledgerline.graph.adjudicator.IntegrityError", FakeIntegrityError)
    return extracted


def _report(db) -> Report:
    co = Company(cik="0000789019", ticker="MSFT", name="Microsoft")
    db.add(co)
    db.flush()
    db.add(Filing(company_id=co.id, accession_no="0000789019-25-000001", form_type="8-K",
                  filed_at=date(2025, 7, 30), fiscal_year=2025, fiscal_period="Q3",
                  doc_url="u", storage_path="objects/x", source_url="u", **PROV))
    r = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q3")
    db.add(r)
    db.flush()
    return r


@pytest.mark.db
def test_run_report_extracts_verifies_and_ranks(db, monkeypatch):
    extracted = _install_fakes(monkeypatch)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "done" and r.error is None and len(extracted) == 1
    assert r.cost_usd == pytest.approx(0.04) and r.latency_s is not None
    verdicts = db.scalars(select(Verdict).where(Verdict.report_id == r.id)
                          .order_by(Verdict.severity.desc())).all()
    assert [v.label for v in verdicts] == ["contradicted", "unverifiable"]
    assert verdicts[0].severity > verdicts[1].severity > 0
    ev = db.scalars(select(Evidence).where(Evidence.report_id == r.id)).all()
    assert sorted(e.agent for e in ev) == ["critic", "critic", "xbrl_table", "xbrl_table"]
    xbrl = next(e for e in ev if e.agent == "xbrl_table")
    assert (xbrl.model, xbrl.input_tokens, xbrl.cache_read_tokens) == ("claude-sonnet-5", 1000, 800)
    assert xbrl.trace_id == next(v.trace_id for v in verdicts if v.claim_id == xbrl.claim_id)
    assert xbrl.latency_ms >= 2  # B4: read from the `agent.xbrl_table` span
    names = db.scalars(select(Span.name).where(Span.trace_id == verdicts[0].trace_id)).all()
    assert sorted(names) == ["agent.xbrl_table", "verify_claim"]  # one verify_claim span
    assert "heartbeat_at" in r.config

    # A second run neither re-extracts nor re-verifies.
    run_report(db, r.id)
    assert len(extracted) == 1 and r.status == "done"
    assert r.cost_usd == pytest.approx(0.04)


@pytest.mark.db
def test_failure_keeps_finished_claims_and_marks_failed(db, monkeypatch):
    _install_fakes(monkeypatch, fail_on=2)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "failed" and r.error == "RuntimeError"  # class only: no traceback, no internals
    assert r.finished_at is not None
    assert len(db.scalars(select(Verdict).where(Verdict.report_id == r.id)).all()) == 1
    assert r.cost_usd == pytest.approx(0.02)  # B5: the committed claim's spend survives

    # Resuming verifies only the claim that failed.
    _install_fakes(monkeypatch)
    run_report(db, r.id)
    assert r.status == "done"
    assert len(db.scalars(select(Verdict).where(Verdict.report_id == r.id)).all()) == 2


@pytest.mark.db
def test_quarter_without_sources_fails_with_reason(db, monkeypatch):
    _install_fakes(monkeypatch)
    r = _report(db)
    r.fiscal_period = "Q1"  # the only 8-K is Q3
    db.flush()
    run_report(db, r.id)
    assert r.status == "failed" and "no claims for FY2025 Q1" in r.error


@pytest.mark.db
def test_integrity_failure_is_one_refused_claim_not_a_failed_report(db, monkeypatch):
    _install_fakes(monkeypatch, integrity_fail_on=1)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "done"
    verdicts = db.scalars(select(Verdict).where(Verdict.report_id == r.id)
                          .order_by(Verdict.claim_id)).all()
    bad = verdicts[0]
    assert (bad.label, bad.refused, bad.confidence, bad.citations) == ("unverifiable", True, 0, [])
    assert "97.1" not in bad.rationale  # the failed text and its numbers are not shown
    assert bad.cost_usd == pytest.approx(0.005)  # spend recorded on the finished agent span
    assert r.cost_usd == pytest.approx(0.025)
    names = db.scalars(select(Span.name).where(Span.trace_id == bad.trace_id)).all()
    assert sorted(names) == ["agent.xbrl_table", "verify_claim"]


def test_deviation_reads_the_agent_payload_magnitude():
    from ledgerline.pipeline import _deviation

    def res(**payload):
        return types.SimpleNamespace(evidence=[
            EvidenceItem(agent="xbrl_table", stance="contradicts", summary="x", payload=payload,
                         citations=[Citation(kind="xbrl_fact", ref_id=1, value=1.0)])])
    assert _deviation(res(deviation=-0.03)) == pytest.approx(0.03)
    assert _deviation(res(relative_error=0.5)) is None  # the old key is not read


@pytest.mark.db
def test_fy_report_fails_with_reason(db, monkeypatch):
    _install_fakes(monkeypatch)
    r = _report(db)
    r.fiscal_period = "FY"
    db.flush()
    run_report(db, r.id)
    assert r.status == "failed" and "request Q4" in r.error


def test_cli_rejects_fy(capsys):
    from ledgerline.pipeline import main

    with pytest.raises(SystemExit) as e:
        main(["--ticker", "AAPL", "--fy", "2025", "--fp", "FY"])
    assert e.value.code == 2 and "request Q4" in capsys.readouterr().err

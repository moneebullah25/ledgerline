"""run_report against fake extraction and graph modules (the real ones are built separately
and exercised by their own tests); the database side is real."""

import sys
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


def _install_fakes(monkeypatch, *, fail_on: int | None = None) -> list[int]:
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
        with tracer.span("xbrl_table"):
            pass
        fact = Citation(kind="xbrl_fact", ref_id=1, value=94.0e9, unit="USD")
        contradicted = claim.hedging_score == 0.0
        return FakeResult(
            verdict=VerdictDraft(
                label="contradicted" if contradicted else "unverifiable",
                confidence=0.9 if contradicted else 0.5, rationale="See the filing.",
                citations=[fact] if contradicted else []),
            evidence=[EvidenceItem(agent="xbrl_table", stance="contradicts",
                                   summary="lower", citations=[fact])],
            critic=CriticOutput(counter_stance="supports", argument="Rounding?"),
            refused=False, prompt_version="p1", model="claude-opus-5", cost_usd=0.02,
            usages=[("xbrl_table", Usage("claude-sonnet-5", 1000, 100, 800, 0))],
        )

    monkeypatch.setitem(sys.modules, "ledgerline.extract.pipeline", types.SimpleNamespace(
        EXTRACTOR_VERSION="fake-1", extract_for_filing=extract_for_filing,
        extract_for_call=lambda s, cid: 0))
    monkeypatch.setitem(sys.modules, "ledgerline.graph.build",
                        types.SimpleNamespace(verify_claim=verify_claim))
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
    names = db.scalars(select(Span.name).where(Span.trace_id == verdicts[0].trace_id)).all()
    assert sorted(names) == ["verify_claim", "xbrl_table"]

    # A second run neither re-extracts nor re-verifies.
    run_report(db, r.id)
    assert len(extracted) == 1 and r.status == "done"
    assert r.cost_usd == pytest.approx(0.04)


@pytest.mark.db
def test_failure_keeps_finished_claims_and_marks_failed(db, monkeypatch):
    _install_fakes(monkeypatch, fail_on=2)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "failed" and "graph exploded" in r.error
    assert r.finished_at is not None
    assert len(db.scalars(select(Verdict).where(Verdict.report_id == r.id)).all()) == 1

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

"""run_report against fake extraction and graph modules (the real ones are built separately
and exercised by their own tests); the database side is real."""

import sys
import threading
import time
import types
from datetime import date

import pytest
import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_report_generate import PROV

from ledgerline import llm
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


REFINE = Usage("claude-haiku-4-5", 1000, 100, 0, 800)  # the extraction's LLM refine batch


def _install_fakes(monkeypatch, *, fail_on: int | None = None,
                   integrity_fail_on: int | None = None,
                   error: Exception | None = None, refine: Usage | None = None,
                   offline_flags: list[bool] | None = None) -> list[int]:
    extracted: list[int] = []

    def extract_for_filing(s, filing_id, on_progress=None):
        extracted.append(filing_id)
        if refine is not None:
            on_progress(refine)
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

    def verify_claim(s, claim, tracer, offline=False):
        calls["n"] += 1
        if offline_flags is not None:
            offline_flags.append(offline)
        if calls["n"] == fail_on:
            raise error or OperationalError("SELECT 1", {}, Exception("db gone"))
        with tracer.span("verify_claim", claim_id=claim.id):  # as graph.build does
            with tracer.span("agent.xbrl_table") as a:
                a["cost_usd"] = 0.005
                time.sleep(0.002)
            if calls["n"] == integrity_fail_on:
                err = FakeIntegrityError("adjudicator output failed audit twice: [97.1]")
                err.usage = types.SimpleNamespace(cost_usd=0.001)
                raise err
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
            usages=[("xbrl_table", Usage("claude-sonnet-5", 1000, 100, 800, 50))],
        )

    monkeypatch.setitem(sys.modules, "ledgerline.extract.pipeline", types.SimpleNamespace(
        extractor_version=lambda: "fake-1", extract_for_filing=extract_for_filing,
        extract_for_call=lambda s, cid, on_progress=None: 0))
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
    assert (xbrl.model, xbrl.input_tokens, xbrl.cache_read_tokens, xbrl.cache_write_tokens) == (
        "claude-sonnet-5", 1000, 800, 50)
    assert xbrl.trace_id == next(v.trace_id for v in verdicts if v.claim_id == xbrl.claim_id)
    assert xbrl.latency_ms >= 2  # B4: read from the `agent.xbrl_table` span
    names = db.scalars(select(Span.name).where(Span.trace_id == verdicts[0].trace_id)).all()
    assert sorted(names) == ["agent.xbrl_table", "verify_claim"]  # one verify_claim span
    assert "heartbeat_at" in r.config and r.config["mode"] == "offline"  # conftest: no LLM

    # A second run neither re-extracts nor re-verifies.
    run_report(db, r.id)
    assert len(extracted) == 1 and r.status == "done"
    assert r.cost_usd == pytest.approx(0.04)


@pytest.mark.db
def test_failure_keeps_finished_claims_and_marks_failed(db, monkeypatch):
    _install_fakes(monkeypatch, fail_on=2)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "failed" and r.error == "OperationalError"  # class only, no internals
    assert r.finished_at is not None
    assert len(db.scalars(select(Verdict).where(Verdict.report_id == r.id)).all()) == 1
    assert r.cost_usd == pytest.approx(0.02)  # B5: the committed claim's spend survives

    # Resuming verifies only the claim that failed.
    _install_fakes(monkeypatch)
    run_report(db, r.id)
    assert r.status == "done"
    assert len(db.scalars(select(Verdict).where(Verdict.report_id == r.id)).all()) == 2


@pytest.mark.db
@pytest.mark.parametrize("err", [llm.RateLimited("claude-opus-5: rate-limited"),
                                 llm.LLMUnavailable("claude-opus-5: CLI is not authenticated")])
def test_llm_infra_error_fails_the_report_not_the_claim(db, monkeypatch, err):
    """K9: an unreachable model would fail every claim the same way, so the report fails
    (and can be POSTed again) instead of storing a no-verdict per claim."""
    _install_fakes(monkeypatch, fail_on=2, error=err)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "failed" and r.error == f"{type(err).__name__}: {err}"
    verdicts = db.scalars(select(Verdict).where(Verdict.report_id == r.id)).all()
    assert len(verdicts) == 1 and not verdicts[0].refused  # the claim before it is kept


@pytest.mark.db
def test_refine_spend_is_part_of_the_report_cost(db, monkeypatch):
    """K8: the extraction's Haiku refine usage (cache writes included) counts."""
    _install_fakes(monkeypatch, refine=REFINE)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "done"
    assert r.config["extract_cost_usd"] == pytest.approx(REFINE.cost_usd)
    assert r.cost_usd == pytest.approx(0.04 + REFINE.cost_usd)


@pytest.mark.db
def test_budget_guard_switches_later_claims_offline(db, monkeypatch):
    """K7: claim 1 spends 0.02 against a pace of 0.03 x 1/2, so claim 2 runs offline."""
    flags: list[bool] = []
    _install_fakes(monkeypatch, offline_flags=flags)
    monkeypatch.setattr("ledgerline.pipeline.COST_BUDGET_USD", 0.03)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "done" and flags == [False, True]
    assert r.config["budget_guard"] == {"after_claims": 1, "of": 2, "spent_usd": 0.02}
    from ledgerline.report.generate import build_report

    assert any("budget guard" in w for w in build_report(db, r.id).warnings)


@pytest.mark.db
def test_budget_guard_stays_off_within_pace(db, monkeypatch):
    flags: list[bool] = []
    _install_fakes(monkeypatch, offline_flags=flags)
    r = _report(db)
    run_report(db, r.id)
    assert flags == [False, False] and "budget_guard" not in r.config


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
    assert "citation-integrity" in bad.rationale
    assert bad.cost_usd == pytest.approx(0.006)  # the finished agent span + err.usage
    assert r.cost_usd == pytest.approx(0.026)
    names = db.scalars(select(Span.name).where(Span.trace_id == bad.trace_id)).all()
    assert sorted(names) == ["agent.xbrl_table", "verify_claim"]


@pytest.mark.db
@pytest.mark.parametrize("err", [RuntimeError("model refused"), ValueError("wiring")])
def test_any_claim_error_is_a_refused_verdict_naming_the_class(db, monkeypatch, err):
    _install_fakes(monkeypatch, fail_on=1, error=err)
    r = _report(db)
    run_report(db, r.id)
    assert r.status == "done"
    bad = db.scalars(select(Verdict).where(Verdict.report_id == r.id)
                     .order_by(Verdict.claim_id)).first()
    assert (bad.label, bad.refused) == ("unverifiable", True)
    assert bad.rationale == f"No verdict: {type(err).__name__} while verifying the claim."
    assert str(err) not in bad.rationale


@pytest.mark.db
def test_heartbeat_does_not_clobber_attempts_written_elsewhere(db, monkeypatch):
    _install_fakes(monkeypatch)
    r = _report(db)
    r.config = {"attempts": 1}
    db.flush()
    # The worker (another session) bumps attempts behind this session's in-memory dict.
    db.execute(sa.text("UPDATE report SET config = config || '{\"attempts\": 2}' WHERE id = :id"),
               {"id": r.id})
    run_report(db, r.id)
    assert r.status == "done" and r.config["attempts"] == 2


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


@pytest.fixture
def committed_report(db_url):
    """A committed company/8-K/report (worker Sessions are separate connections and cannot
    see the rolled-back `db` fixture's rows), deleted again afterwards."""
    engine = sa.create_engine(db_url)
    with Session(engine) as s:
        r = _report(s)
        s.commit()
        ids = (r.id, r.company_id)
    try:
        yield engine, ids[0]
    finally:
        with engine.begin() as c:
            traces = select(Verdict.trace_id).where(Verdict.report_id == ids[0])
            c.execute(sa.delete(Span).where(Span.trace_id.in_(traces)))
            c.execute(sa.delete(Report).where(Report.id == ids[0]))  # cascades evidence, verdicts
            c.execute(sa.delete(Claim).where(Claim.company_id == ids[1]))
            c.execute(sa.delete(Filing).where(Filing.company_id == ids[1]))
            c.execute(sa.delete(Company).where(Company.id == ids[1]))
        engine.dispose()


@pytest.mark.db
def test_claims_verify_in_parallel_on_their_own_sessions(committed_report, monkeypatch):
    """K6: with an Engine-bound Session both claims are in flight at once, each on a Session
    other than the caller's; the extraction heartbeat (J10) commits on its own connection."""
    engine, report_id = committed_report
    _install_fakes(monkeypatch, refine=REFINE)
    fake = sys.modules["ledgerline.graph.build"].verify_claim
    extract = sys.modules["ledgerline.extract.pipeline"].extract_for_filing
    barrier, sessions, beats = threading.Barrier(2, timeout=10), [], []

    def verify_claim(s, claim, tracer, offline=False):
        sessions.append(s)
        barrier.wait()  # times out (BrokenBarrierError -> no verdict) unless both run at once
        return fake(s, claim, tracer, offline)

    def beat() -> str:
        with engine.connect() as c:  # another connection sees only committed heartbeats
            return c.scalar(select(Report.config["heartbeat_at"].astext)
                            .where(Report.id == report_id))

    def extract_for_filing(s, filing_id, on_progress=None):
        beats.append(beat())
        n = extract(s, filing_id, on_progress)  # calls on_progress once
        beats.append(beat())
        return n

    monkeypatch.setattr(sys.modules["ledgerline.graph.build"], "verify_claim", verify_claim)
    monkeypatch.setattr(sys.modules["ledgerline.extract.pipeline"], "extract_for_filing",
                        extract_for_filing)
    monkeypatch.setattr("ledgerline.pipeline.VERIFY_WORKERS", 2)
    with Session(engine, expire_on_commit=False) as s:
        run_report(s, report_id)
        r = s.get(Report, report_id)
        assert r.status == "done", r.error
        verdicts = s.scalars(select(Verdict).where(Verdict.report_id == report_id)).all()
        assert sorted(v.label for v in verdicts) == ["contradicted", "unverifiable"]
        assert len(sessions) == 2 and s not in sessions and sessions[0] is not sessions[1]
        assert beats[1] > beats[0]  # committed mid-extraction by on_progress
        assert r.cost_usd == pytest.approx(0.04 + REFINE.cost_usd)


def test_verify_claim_offline_never_calls_a_model(monkeypatch):
    """K7: the budget guard's offline=True keeps the critic and adjudicator deterministic
    even with credentials, where needs_judgment would otherwise call the models."""
    from threading import Lock

    from ledgerline.graph import build
    from ledgerline.graph.state import Ctx
    from ledgerline.tracing import Tracer

    claim = Claim(id=7, company_id=1, fiscal_year=2025, fiscal_period="Q1",
                  text="revenue of $94.9 billion", span_start=0, span_end=1,
                  claim_type="reported_result", filing_id=1, metric="revenue", value=94.9e9,
                  unit="USD", target_fiscal_year=2025, target_fiscal_period="Q1")
    fact = Citation(kind="xbrl_fact", ref_id=11, value=94_930e6, unit="USD")
    evidence = {
        "xbrl_table": EvidenceItem(agent="xbrl_table", stance="supports", summary="ok",
                                   citations=[fact], payload={"periods": [[2025, "Q1"]]}),
        "filing_rag": EvidenceItem(agent="filing_rag", stance="contradicts", summary="no",
                                   citations=[Citation(kind="chunk", ref_id=3, quote="q")],
                                   payload={"margin": 0.5}),
    }
    monkeypatch.setattr(build, "context", lambda s, c: Ctx(1, "1", "AAPL", None, None, Lock()))
    agents = {name: (lambda s, c, ctx, lock, e=e: (e, llm.OFFLINE)) for name, e in evidence.items()}
    monkeypatch.setattr(llm, "available", lambda: True)

    def no_model(*a, **k):
        raise AssertionError("a model was called")

    monkeypatch.setattr(llm, "structured", no_model)
    r = build.verify_claim(None, claim, Tracer(), agents=agents, offline=True)
    assert r.model is None and r.cost_usd == 0 and r.verdict.label == "supported"

import re
import uuid
from datetime import UTC, date, datetime

import pytest

from ledgerline.models import (
    Chunk,
    Claim,
    Company,
    Evidence,
    Filing,
    MarketPrice,
    Report,
    Span,
    Verdict,
    XbrlFact,
)
from ledgerline.report.generate import build_report, to_markdown
from ledgerline.schemas import Citation, uncited_numbers

NOW = datetime(2025, 8, 1, tzinfo=UTC)
PROV = {"fetched_at": NOW, "sha256": "0" * 64, "license_note": "public SEC filing"}


def seed_report(s, n_claims: int = 3, ticker: str = "AAPL") -> Report:
    """A finished report: n claims over one 8-K, each with evidence, a verdict and a span.
    The first claim's rationale is contradicted and cites a fact, a chunk and a price."""
    co = Company(cik="0000320193", ticker=ticker, name="Apple Inc.", fiscal_year_end="0927")
    s.add(co)
    s.flush()
    ten_q = Filing(company_id=co.id, accession_no="0000320193-25-000073", form_type="10-Q",
                   filed_at=date(2025, 8, 1), fiscal_year=2025, fiscal_period="Q3",
                   doc_url="https://www.sec.gov/aapl-10q.htm", storage_path="objects/a",
                   source_url="https://www.sec.gov/aapl-10q.htm", **PROV)
    release = Filing(company_id=co.id, accession_no="0000320193-25-000071", form_type="8-K",
                     filed_at=date(2025, 7, 31), fiscal_year=2025, fiscal_period="Q3",
                     doc_url="https://www.sec.gov/ex99.htm", storage_path="objects/b",
                     source_url="https://www.sec.gov/ex99.htm", **PROV)
    s.add_all([ten_q, release])
    s.flush()
    fact = XbrlFact(company_id=co.id, tag="us-gaap:Revenues", unit="USD", value=94036000000,
                    period_start=date(2025, 3, 30), period_end=date(2025, 6, 28),
                    duration_days=90, fiscal_year=2025, fiscal_period="Q3", form="10-Q",
                    accession_no=ten_q.accession_no, filed=date(2025, 8, 1),
                    source_url="https://data.sec.gov/companyfacts.json", **PROV)
    chunk = Chunk(filing_id=ten_q.id, chunk_index=0, section_path="Part I > Item 2. MD&A",
                  char_start=0, char_end=60, token_count=12,
                  text="Total net sales were $94,036 million for the quarter.")
    price = MarketPrice(ticker=ticker, day=date(2025, 8, 1), close=202.38, adj_close=202.38,
                        source_url="https://example.com/prices", **PROV)
    report = Report(company_id=co.id, fiscal_year=2025, fiscal_period="Q3", status="done",
                    latency_s=12.5, cost_usd=0.41, finished_at=NOW, config={"mode": "offline"})
    s.add_all([fact, chunk, price, report])
    s.flush()
    for i in range(n_claims):
        claim = Claim(company_id=co.id, fiscal_year=2025, fiscal_period="Q3",
                      filing_id=release.id, span_start=i * 100, span_end=i * 100 + 50,
                      text="Revenue was $97.0 billion for the June quarter.",
                      claim_type="reported_result", metric="revenue", value=97.0e9,
                      unit="USD", hedging_score=0.1, extractor_version="test")
        s.add(claim)
        s.flush()
        trace = uuid.uuid4()
        cites = [
            Citation(kind="xbrl_fact", ref_id=fact.id, value=94036000000.0, unit="USD"),
            Citation(kind="chunk", ref_id=chunk.id, quote="Total net sales were $94,036 million"),
            Citation(kind="price", ref_id=price.id),
        ]
        first = i == 0
        s.add(Evidence(report_id=report.id, claim_id=claim.id, agent="xbrl_table",
                       stance="contradicts", summary="Filed revenue is lower.",
                       citations=[c.model_dump() for c in cites[:1]],
                       payload={"deviation": -0.0315}, trace_id=trace))
        s.add(Verdict(
            report_id=report.id, claim_id=claim.id,
            label="contradicted" if first else "supported", confidence=0.9 if first else 0.8,
            rationale="The 10-Q reports revenue of $94,036 million, not $97.0 billion."
            if first else "Consistent with the filing.",
            citations=[c.model_dump() for c in (cites if first else cites[:1])],
            severity=1.8 if first else 0.0, prompt_version="v1", model=None, trace_id=trace,
        ))
        s.add(Span(trace_id=trace, name="verify_claim", started_at=NOW, ended_at=NOW,
                   attrs={"latency_ms": 5}))
    s.flush()
    return report


def _visible_numbers_text(md: str) -> str:
    """Drop code spans (scores, identifiers) and link targets: those are not figures."""
    return re.sub(r"\]\([^)]*\)", "]", re.sub(r"`[^`]*`", "", md))


@pytest.mark.db
def test_report_is_ranked_resolved_and_fully_cited(db):
    report = seed_report(db)
    view = build_report(db, report.id)
    assert [f.label for f in view.claims] == ["contradicted", "supported", "supported"]
    top = view.claims[0]
    kinds = {x.kind: x for x in top.sources}
    assert kinds["xbrl_fact"].url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000073/")
    assert "us-gaap:Revenues, FY2025 Q3" in kinds["xbrl_fact"].label
    assert kinds["chunk"].label == "10-Q filed 2025-08-01, Part I > Item 2. MD&A"
    assert kinds["price"].value == 202.38
    assert all(f.uncited_numbers == [] for f in view.claims)

    # NFR-8 over the rendered report: every figure a reader sees traces to a citation.
    md = to_markdown(view)
    cites = [Citation(kind=x.kind, ref_id=x.ref_id, value=x.value, quote=x.quote)
             for f in view.claims for x in f.sources]
    claim_texts = " ".join(f.text for f in view.claims)
    assert uncited_numbers(_visible_numbers_text(md), cites, claim_texts) == []


@pytest.mark.db
def test_uncited_rationale_is_flagged(db):
    report = seed_report(db, n_claims=1)
    v = db.query(Verdict).filter_by(report_id=report.id).one()
    v.rationale = "Revenue was really $91,200 million."
    db.flush()
    finding = build_report(db, report.id).claims[0]
    assert finding.uncited_numbers == [91200.0]
    assert "WARNING uncited" in to_markdown(build_report(db, report.id))


@pytest.mark.db
def test_unknown_report(db):
    assert build_report(db, uuid.uuid4()) is None


@pytest.mark.db
def test_revised_row_keeps_the_cited_value_and_shows_the_revision(db):
    """F1: the verdict was judged on the cited value; an upsert since then is shown, not
    silently swapped in (the audit accepts the cited value, never an uncited one)."""
    report = seed_report(db, n_claims=1)
    v = db.query(Verdict).filter_by(report_id=report.id).one()
    fact_id = v.citations[0]["ref_id"]
    v.citations = [Citation(kind="xbrl_fact", ref_id=fact_id, value=91.2e9, unit="USD")
                   .model_dump()]
    v.rationale = "Filed revenue was $91.2 billion, not $90 billion."
    db.flush()
    view = build_report(db, report.id)
    src = view.claims[0].sources[0]
    assert (src.value, src.revised) == (91.2e9, 94036000000.0)
    assert view.claims[0].uncited_numbers == [90.0]
    assert "91,200,000,000 USD (since revised: 94,036,000,000 USD)" in to_markdown(view)


@pytest.mark.db
def test_computed_citation_is_rendered_and_citable(db):
    report = seed_report(db, n_claims=1)
    v = db.query(Verdict).filter_by(report_id=report.id).one()
    v.citations = v.citations[:1] + [Citation(
        kind="computed", ref_id="revenue:2025Q3:yoy_change", value=5.4, unit="percent",
    ).model_dump()]
    v.rationale = "Revenue grew 5.4% year over year."
    db.flush()
    view = build_report(db, report.id)
    src = view.claims[0].sources[1]
    assert src.kind == "computed" and src.value == 5.4 and src.url is None
    assert src.label.startswith("computed from the cited facts")
    assert view.claims[0].uncited_numbers == []
    assert "computed from the cited facts (revenue:2025Q3:yoy_change): 5.4 percent" in (
        to_markdown(view))


@pytest.mark.db
def test_offline_mode_and_budget_warnings_are_shown(db):
    report = seed_report(db, n_claims=1)  # model=None: the offline adjudicator
    report.cost_usd, report.latency_s = 3.5, 1500.0
    db.flush()
    view = build_report(db, report.id)
    assert view.mode == "offline (no LLM credentials)"
    assert len(view.warnings) == 2
    md = to_markdown(view)
    assert "mode: offline (no LLM credentials)" in md
    assert "over the 3.00 USD budget" in md and "over the 20 min budget" in md

    report.config = {"mode": "online"}  # recorded at run time, not inferred from verdicts
    report.cost_usd, report.latency_s = 0.4, 60.0
    db.flush()
    view = build_report(db, report.id)
    assert (view.mode, view.warnings) == ("online", [])


@pytest.mark.db
def test_findings_grouped_by_label_and_no_verdict_rendered(db):
    """F7: contradicted, then unverifiable, then supported, whatever the severity.
    F6: a claim whose verification raised says so, not "refused: no evidence"."""
    report = seed_report(db, n_claims=4)
    vs = db.query(Verdict).filter_by(report_id=report.id).order_by(Verdict.claim_id).all()
    vs[1].label, vs[1].severity = "supported", 5.0
    vs[2].label, vs[2].refused, vs[2].severity = "unverifiable", True, 0.0
    vs[2].rationale = "No evidence."
    vs[3].label, vs[3].refused, vs[3].severity = "unverifiable", True, 0.0
    vs[3].rationale = "No verdict: RuntimeError while verifying the claim."
    vs[3].citations = []
    db.flush()
    view = build_report(db, report.id)
    assert [f.label for f in view.claims] == [
        "contradicted", "unverifiable", "unverifiable", "supported"]
    md = to_markdown(view)
    assert "unverifiable (no verdict: RuntimeError while verifying the claim)" in md
    assert md.count("(refused: no evidence)") == 1

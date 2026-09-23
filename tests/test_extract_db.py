from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from ledgerline.extract.pipeline import EXTRACTOR_VERSION, extract_for_filing, link_across_quarters
from ledgerline.models import Claim, Company, Filing

FIX = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.db


def _filing(s, company, acc, fy, fp):
    f = Filing(company_id=company.id, accession_no=acc, form_type="8-K", filed_at=date(2026, 8, 26),
               fiscal_year=fy, fiscal_period=fp, doc_url="https://example.com", storage_path="objects/x.htm",
               source_url="https://example.com", fetched_at=datetime.now(UTC), sha256="0" * 64,
               license_note="test")
    s.add(f)
    s.flush()
    return f


def _company(s):
    c = Company(cik="0001045810", ticker="NVDA", name="NVIDIA", fiscal_year_end="0131")
    s.add(c)
    s.flush()
    return c


def test_extract_for_filing_is_idempotent(db, monkeypatch):
    text = (FIX / "ex991_nvda_q2fy27.txt").read_text()
    monkeypatch.setattr("ledgerline.ingest.documents.load_text", lambda f: text)
    c = _company(db)
    f = _filing(db, c, "0001045810-26-000073", 2027, "Q2")
    n = extract_for_filing(db, f.id, use_ner=False)
    assert n > 0
    assert extract_for_filing(db, f.id, use_ner=False) == 0  # unique span constraint
    rows = db.scalars(select(Claim).where(Claim.filing_id == f.id)).all()
    assert len(rows) == n
    assert all(r.speaker_role == "executive" and r.extractor_version == EXTRACTOR_VERSION
               and (r.fiscal_year, r.fiscal_period) == (2027, "Q2") for r in rows)
    assert all(text[r.span_start:r.span_end] == r.text for r in rows)


def test_link_guidance_to_later_actual(db):
    c = _company(db)
    q1 = _filing(db, c, "a1", 2027, "Q1")
    q2 = _filing(db, c, "a2", 2027, "Q2")
    common = {"company_id": c.id, "text": "x", "extractor_version": EXTRACTOR_VERSION, "unit": "USD"}

    def add(f, start, **kw):
        row = Claim(fiscal_year=f.fiscal_year, fiscal_period=f.fiscal_period, filing_id=f.id,
                    span_start=start, span_end=start + 1, **common, **kw)
        db.add(row)
        db.flush()
        return row

    guide = add(q1, 0, claim_type="guidance", metric="revenue", value=89e9, value_high=93e9,
                target_fiscal_year=2027, target_fiscal_period="Q2")
    add(q1, 5, claim_type="guidance", metric="revenue", value=95e9,  # a later quarter's target
        target_fiscal_year=2027, target_fiscal_period="Q3")
    actual = add(q2, 0, claim_type="reported_result", metric="revenue", value=96e9,
                 target_fiscal_year=2027, target_fiscal_period="Q2")
    other = add(q2, 5, claim_type="reported_result", metric="net_income", value=5e9,
                target_fiscal_year=2027, target_fiscal_period="Q2")
    same_q = add(q1, 9, claim_type="reported_result", metric="revenue", value=81e9,
                 target_fiscal_year=2027, target_fiscal_period="Q2")  # not earlier than itself

    assert link_across_quarters(db, c.id) == 1
    db.expire_all()
    assert actual.linked_claim_id == guide.id
    assert other.linked_claim_id is None and same_q.linked_claim_id is None
    assert link_across_quarters(db, c.id) == 1  # re-run: same result, no duplicates

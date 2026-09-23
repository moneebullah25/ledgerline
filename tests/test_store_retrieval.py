import re
import sys
import types
import zlib
from datetime import UTC, date, datetime

import numpy as np
import pytest
from sqlalchemy import func, select

from ledgerline.models import Chunk, Company, Filing
from ledgerline.store import embed
from ledgerline.store.index import index_filing
from ledgerline.store.retrieval import _or_query, rrf, search


def test_rrf_math():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [d for d, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1][1] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2][1] == pytest.approx(1 / 62)


def test_or_query_keeps_numbers_and_drops_operators():
    assert _or_query('Net sales "94,930" -or 94.9?') == "Net or sales or 94,930 or 94.9"


def _bow(text: str) -> np.ndarray:
    """Stub embedding: hashed bag of words, so the db test needs no model download."""
    v = np.zeros(384, dtype=np.float32)
    for w in re.findall(r"\w+", text.lower()):
        v[zlib.crc32(w.encode()) % 384] += 1
    return v / (np.linalg.norm(v) or 1)


@pytest.mark.db
def test_index_and_search(db, monkeypatch):
    doc = ("PART I\nItem 2. Management's Discussion and Analysis\n"
           "Total net sales were 94,930 million, driven by iPhone.\n"
           "PART II\nItem 1A. Risk Factors\nSupply chain disruption could hurt margins.\n")
    monkeypatch.setitem(sys.modules, "ledgerline.ingest.documents",
                        types.SimpleNamespace(load_text=lambda f: doc))
    monkeypatch.setattr(embed, "embed_passages", lambda ts: [_bow(t) for t in ts])
    monkeypatch.setattr(embed, "embed_query", _bow)

    co = Company(cik="0000000001", ticker="TST", name="Test Co")
    db.add(co)
    db.flush()
    prov = dict(source_url="x", fetched_at=datetime.now(UTC), sha256="0" * 64, license_note="t")
    f = Filing(company_id=co.id, accession_no="0000000001-26-000001", form_type="10-Q",
               filed_at=date(2026, 7, 31), doc_url="x", storage_path="x", **prov)
    db.add(f)
    db.flush()

    assert index_filing(db, f.id) == 2
    assert index_filing(db, f.id) == 2  # idempotent
    assert db.scalar(select(func.count()).select_from(Chunk).where(Chunk.filing_id == f.id)) == 2

    hits = search(db, "What were net sales of 94,930?", company_id=co.id, rerank=False)
    assert hits[0].section_path.endswith("Item 2. Management's Discussion and Analysis")
    assert doc[hits[0].char_start:hits[0].char_end] == hits[0].text
    risk = search(db, "supply chain risks", company_id=co.id, k=1, rerank=False)
    assert risk[0].section_path == "Part II > Item 1A. Risk Factors"
    assert search(db, "net sales", company_id=co.id + 1, rerank=False) == []

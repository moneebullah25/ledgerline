import json
from collections import Counter
from datetime import UTC, date, datetime

import pytest

from ledgerline.evals import golden
from ledgerline.models import Company, Filing


def _claim(**kw) -> dict:
    base = {"text": "Revenue was $94.9 billion", "claim_type": "reported_result", "metric": "revenue",
            "value": 94.9e9, "value_high": None, "unit": "USD", "fiscal_year": 2025,
            "fiscal_period": "Q3", "target_fiscal_year": 2025, "target_fiscal_period": "Q3",
            "comparison": None, "hedging_score": 0.0, "speaker_role": None}
    return base | kw


def _rec(i: str, gold: str = "supported", **claim) -> dict:
    return {"id": i, "gold": gold, "evidence": "x", "pass1": gold, "pass2": None,
            "claim": _claim(**claim), "accession_no": "a", "span": [0, 1],
            "extractor_version": "v"}


# ------------------------------------------------------------------ frozen data files


def test_frozen_golden_set_is_valid_and_documented():
    recs = golden.load()
    assert len(recs) >= 120  # R3 floor; target 200
    assert all(r["annotator"] and r["evidence"] for r in recs)
    assert {r["gold"] for r in recs} <= set(golden.LABELS)
    # Adjudication is recorded wherever the two passes disagree.
    for r in recs:
        if r["pass2"] is not None and r["pass2"] != r["pass1"]:
            assert r["adjudication"]


def test_frozen_heldout_set_is_blind_new_and_stratified():
    held, gold = golden.load(golden.HELDOUT), golden.load()
    assert len(held) >= 100 and all(r["id"].startswith("h") for r in held)
    assert all(r["annotator"] and r["evidence"] for r in held)
    assert {r["extractor_version"] for r in held} == {golden.HELDOUT_EXTRACTOR}
    spans = {(g["accession_no"], tuple(g["span"])) for g in gold}
    for r in held:  # no filing span shared with (or overlapping) a golden record
        assert not any(acc == r["accession_no"] and golden._overlaps(tuple(r["span"]), sp)
                       for acc, sp in spans)
    types = Counter(r["claim"]["claim_type"] for r in held)
    assert set(types) == {"reported_result", "comparison", "guidance", "qualitative"}
    assert len({r["ticker"] for r in held}) == 15
    # Guidance with a known outcome is labelled against it (not left unverifiable).
    assert any(r["claim"]["claim_type"] == "guidance" and r["gold"] != "unverifiable"
               for r in held)


def test_heldout_candidates_drop_golden_spans_and_judged_claims():
    g = [{"accession_no": "a", "span": [10, 20]}]
    claims = [{"source_claim_id": 1, "accession_no": "a", "span": [15, 30]},   # overlaps
              {"source_claim_id": 2, "accession_no": "a", "span": [20, 30]},   # adjacent: kept
              {"source_claim_id": 3, "accession_no": "b", "span": [10, 20]},   # other filing
              {"source_claim_id": 4, "accession_no": "b", "span": [40, 50]}]   # has a verdict
    kept = golden.heldout_candidates(claims, g, judged={4})
    assert [c["source_claim_id"] for c in kept] == [2, 3]


def test_perturbations_come_from_supported_golden_claims():
    base = {r["id"]: r for r in golden.load()}
    for p in golden.load(golden.PERTURBED):
        src = base[p["perturbed_from"]]
        assert p["gold"] == "contradicted" and src["gold"] == "supported"
        assert p["claim"]["text"] != src["claim"]["text"]
        assert abs(p["claim"]["value"] - src["claim"]["value"]) >= 0.09 * abs(src["claim"]["value"])


def test_load_rejects_bad_records(tmp_path):
    f = tmp_path / "g.jsonl"
    for bad in ([_rec("g1"), _rec("g1")], [_rec("g1", gold="false")], [_rec("g1") | {"evidence": ""}],
                [_rec("g1") | {"pass2": "maybe"}]):
        f.write_text("".join(json.dumps(r) + "\n" for r in bad))
        with pytest.raises(ValueError):
            golden.load(f)


# ------------------------------------------------------------------ sampling


def test_stratified_quota_round_robin_and_determinism():
    claims = [{"source_claim_id": i, "ticker": t, "claim_type": ct, "metric": m}
              for i, (t, ct, m) in enumerate(
                  [(t, ct, m) for t in "ABC" for ct in ("reported_result", "guidance")
                   for m in ("revenue", "eps_diluted") for _ in range(3)])]
    quota = {"reported_result": 6, "guidance": 2}
    a = golden.stratified(claims, quota, seed=1)
    assert a == golden.stratified(claims, quota, seed=1)
    assert Counter(c["claim_type"] for c in a) == Counter(quota)
    rr = [c for c in a if c["claim_type"] == "reported_result"]
    assert Counter(c["ticker"] for c in rr) == {"A": 2, "B": 2, "C": 2}  # round-robin
    for t in "ABC":  # metrics interleave within a company
        assert {c["metric"] for c in rr if c["ticker"] == t} == {"revenue", "eps_diluted"}
    assert golden.stratified(claims, {"guidance": 100}, seed=1)  # quota > pool: take all
    assert len(golden.stratified(claims, {"guidance": 100}, seed=1)) == 18


# ------------------------------------------------------------------ number handling


def test_number_matcher_scales_and_rounding():
    match = golden.number_matcher(_claim())
    assert match("Total net sales | 94,930 | 85,777")  # millions within $94.85-94.95B
    assert match("revenue of $94.9 billion")
    assert not match("Total net sales | 95,100")  # outside the printed rounding
    assert not match("Item 9. Controls")
    pct = golden.number_matcher(_claim(text="grew 12%", unit="percent", value=12.0))
    assert pct("Revenue | 12 | 10") and not pct("Revenue | 13 | 10")


def test_rewrite_number_keeps_format_sign_and_skips_tiny_moves():
    text, value = golden._rewrite_number("Revenue | 94,930", 94_930e6, 1.25)
    assert text == "Revenue | 118,662" and value == 118_662e6
    text, value = golden._rewrite_number("EPS declined 42 percent", -42.0, 0.8)
    assert text == "EPS declined 34 percent" and value == -34.0
    assert golden._rewrite_number("sales were up 1%", 1.0, 1.15) is None  # still prints "1"
    assert golden._rewrite_number("no number here", 5.0, 1.25) is None


def test_perturb_only_supported_single_value_results():
    recs = [_rec("g001"), _rec("g002", gold="unverifiable"), _rec("g003", claim_type="guidance"),
            _rec("g004", value_high=99e9), _rec("g005", value=0.0, text="flat")]
    out = golden.perturb(recs, seed=3)
    assert [p["id"] for p in out] == ["p001"]
    assert out[0]["gold"] == "contradicted" and out[0]["perturbed_from"] == "g001"


# ------------------------------------------------------------------ transient Claim


@pytest.mark.db
def test_to_claim_is_transient_and_resolves_filing(db):
    co = Company(cik="0000320193", ticker="AAPL", name="Apple")
    db.add(co)
    db.flush()
    f = Filing(company_id=co.id, accession_no="0000320193-25-000071", form_type="8-K",
               filed_at=date(2025, 7, 31), doc_url="u", storage_path="objects/x.htm", source_url="u",
               fetched_at=datetime.now(UTC), sha256="0" * 64, license_note="t")
    db.add(f)
    db.flush()
    rec = _rec("g001") | {"accession_no": f.accession_no, "span": [10, 20]}
    c = golden.to_claim(db, rec)
    assert c.filing_id == f.id and c.company_id == co.id and float(c.value) == 94.9e9
    assert c not in db and not db.new


@pytest.mark.db
def test_fixture_round_trip_keeps_ids_and_reembeds(db, tmp_path, monkeypatch):
    import numpy as np
    from sqlalchemy import delete, select, text

    from ledgerline.models import Chunk
    from ledgerline.store import embed

    co = Company(cik="0000320193", ticker="AAPL", name="Apple")
    db.add(co)
    db.flush()
    f = Filing(company_id=co.id, accession_no="0000320193-25-000073", form_type="10-Q",
               filed_at=date(2025, 8, 1), fiscal_year=2025, fiscal_period="Q3", doc_url="u",
               storage_path="objects/x.htm", source_url="u", fetched_at=datetime.now(UTC),
               sha256="0" * 64, license_note="t")
    db.add(f)
    db.flush()
    db.add(Chunk(filing_id=f.id, chunk_index=0, section_path="Item 2", char_start=0, char_end=5,
                 text="Total net sales | 94,036", token_count=5))
    db.flush()
    chunk_id = db.scalar(select(Chunk.id))

    sizes = golden.dump_fixture(db, ["AAPL"], tmp_path)
    assert set(sizes) == set(golden._FIXTURE_TABLES) and all(sizes.values())
    db.execute(delete(Chunk))
    db.execute(delete(Filing))
    db.execute(delete(Company))
    monkeypatch.setattr(embed, "embed_passages", lambda texts: np.ones((len(texts), 384)) / 384**0.5)
    assert golden.load_fixture(db, tmp_path) == 1
    row = db.execute(text("SELECT id, embedding IS NOT NULL FROM chunk")).one()
    assert tuple(row) == (chunk_id, True)
    # the id sequence moved past the loaded rows: a new company does not collide
    db.add(Company(cik="0000789019", ticker="MSFT", name="Microsoft"))
    db.flush()

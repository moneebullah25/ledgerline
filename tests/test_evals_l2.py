import pytest

from ledgerline import llm
from ledgerline.evals import gate, l2
from ledgerline.graph import build
from ledgerline.schemas import Citation, EvidenceItem


def test_prf_per_label_and_undefined_cases():
    pairs = [("contradicted", "contradicted"), ("supported", "contradicted"),
             ("supported", "supported"), ("unverifiable", "supported"), ("contradicted", "unverifiable")]
    m = l2.prf(pairs)
    assert m["contradicted"]["precision"] == 0.5 and m["contradicted"]["recall"] == 0.5
    assert m["supported"]["precision"] == 0.5 and m["supported"]["recall"] == 0.5
    assert m["unverifiable"]["precision"] == 0.0 and m["unverifiable"]["f1"] == 0.0
    assert m["accuracy"] == 0.4 and m["confusion"]["supported->contradicted"] == 1
    none = l2.prf([("supported", "supported")])
    assert none["contradicted"]["precision"] is None and none["contradicted"]["recall"] is None
    assert l2.prf([])["accuracy"] is None


def test_error_counts_as_wrong_for_every_label():
    m = l2.prf([("contradicted", l2.ERROR), ("supported", "supported")])
    assert m["contradicted"]["recall"] == 0.0 and m["contradicted"]["precision"] is None


def test_headline_uses_real_set_only():
    real = [{"id": "g1", "gold": "supported"}, {"id": "g2", "gold": "supported"}]
    pert = [{"id": "p1", "gold": "contradicted"}]
    pred = {"g1": {"label": "contradicted"}, "g2": {"label": "supported"},
            "p1": {"label": "contradicted"}}
    m, _ = l2.summarize(real, pert, pred)
    assert m["contradiction_precision"] == 0.0            # real set: one false contradiction
    assert m["combined_contradiction_precision"] == 0.5   # with the perturbation mixed in
    assert m["perturbed_contradiction_recall"] == 1.0


def test_ablation_patches_target_live_attributes():
    ev = EvidenceItem(agent="xbrl_table", stance="supports", summary="s",
                      citations=[Citation(kind="xbrl_fact", ref_id=1, value=1.0, unit="USD")])
    full = {"g1": {"evidence": {"xbrl_table": ev, "filing_rag": ev}}}
    abl = l2.ablations(full)
    assert set(abl) == {"no_reranker", "no_critic", "xbrl_only"}
    for override, ctx in abl.values():
        with ctx:  # patch.object raises if the patched attribute moved
            fns = override("g1")
            assert set(fns) <= set(build.AGENTS)
    only = abl["xbrl_only"][0]("g1")
    assert only["xbrl_table"](None, None, None, None) == (ev, llm.OFFLINE)
    assert only["filing_rag"](None, None, None, None)[0].stance == "insufficient"
    assert set(abl["no_reranker"][0]("g1")) == {"xbrl_table"}  # filing_rag re-runs


def test_run_returns_gate_contract(monkeypatch):
    real = [{"id": "g1", "gold": "contradicted"}, {"id": "g2", "gold": "supported"}]
    monkeypatch.setattr(l2, "load", lambda path: real if path == l2.GOLDEN else [])
    monkeypatch.setattr(l2, "PERTURBED", l2.GOLDEN)  # exists; load() is stubbed anyway
    labels = {"g1": "contradicted", "g2": "supported"}
    monkeypatch.setattr(l2, "predict", lambda s, recs, override=None: {
        r["id"]: {"label": labels[r["id"]], "evidence": {}} for r in recs})
    out = l2.run()
    assert out["metrics"]["contradiction_precision"] == 1.0
    assert out["per_claim"]["g1"] == {"gold": "contradicted", "predicted": "contradicted"}
    for name in ("no_reranker", "no_critic", "xbrl_only"):
        assert f"ablation_{name}_contradiction_precision" in out["metrics"]
    ok, _ = gate.compare(out, {"metrics": {"contradiction_precision": 1.0}})
    assert ok


@pytest.mark.parametrize("label", ["supported", "unverifiable"])
def test_no_predicted_contradiction_makes_headline_none(label):
    m, _ = l2.summarize([{"id": "g1", "gold": "contradicted"}], [], {"g1": {"label": label}})
    assert m["contradiction_precision"] is None and m["contradiction_recall"] == 0.0


@pytest.mark.db
def test_remap_current_prefers_same_type_then_overlap(db):
    from datetime import UTC, date, datetime

    from ledgerline.models import Claim, Company, Filing

    co = Company(cik="0000320193", ticker="AAPL", name="Apple")
    db.add(co)
    db.flush()
    f = Filing(company_id=co.id, accession_no="acc-1", form_type="8-K", filed_at=date(2025, 7, 31),
               doc_url="u", storage_path="objects/x.htm", source_url="u",
               fetched_at=datetime.now(UTC), sha256="0" * 64, license_note="t")
    db.add(f)
    db.flush()

    def add(a, b, claim_type, comparison=None):
        c = Claim(company_id=co.id, filing_id=f.id, fiscal_year=2025, fiscal_period="Q3",
                  span_start=a, span_end=b, text="t" * (b - a), claim_type=claim_type,
                  comparison=comparison, metric="revenue", value=5.0, unit="percent",
                  extractor_version="s4-rules-2")
        db.add(c)
        db.flush()
        return c

    level = add(10, 30, "reported_result")      # exact span, wrong type
    change = add(10, 45, "comparison", "yoy")   # same type and comparison
    add(100, 120, "comparison", "yoy")          # no overlap
    old = {"claim_type": "comparison", "comparison": "yoy"}
    recs = [{"id": "g1", "gold": "supported", "accession_no": "acc-1", "span": [10, 30],
             "claim": old},
            {"id": "g2", "gold": "supported", "accession_no": "acc-1", "span": [10, 30],
             "claim": {"claim_type": "reported_result", "comparison": None}},
            {"id": "g3", "gold": "unverifiable", "accession_no": "acc-1", "span": [60, 80],
             "claim": old}]
    out, unmatched = l2.remap_current(db, recs)
    assert unmatched == ["g3"]
    assert [(r["id"], r["source_claim_id"], r["span"]) for r in out] == [
        ("g1", change.id, [10, 45]), ("g2", level.id, [10, 30])]
    assert out[0]["gold"] == "supported" and out[0]["extractor_version"] == "s4-rules-2"
    assert out[0]["claim"]["value"] == 5.0  # a JSON float, as to_claim expects

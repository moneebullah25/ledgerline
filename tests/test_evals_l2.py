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

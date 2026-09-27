import pytest
from sqlalchemy.exc import OperationalError

pytest.importorskip("scipy")  # [ml] extra — l2.clopper_pearson needs scipy.stats.beta
from ledgerline import llm
from ledgerline.evals import gate, l2
from ledgerline.graph import build
from ledgerline.graph.adjudicator import IntegrityError
from ledgerline.schemas import Citation, EvidenceItem


def test_prf_per_label_and_undefined_cases():
    pairs = [
        ("contradicted", "contradicted"),
        ("supported", "contradicted"),
        ("supported", "supported"),
        ("unverifiable", "supported"),
        ("contradicted", "unverifiable"),
    ]
    m = l2.prf(pairs)
    assert m["contradicted"]["precision"] == 0.5 and m["contradicted"]["recall"] == 0.5
    assert m["supported"]["precision"] == 0.5 and m["supported"]["recall"] == 0.5
    assert m["unverifiable"]["precision"] == 0.0 and m["unverifiable"]["f1"] == 0.0
    assert m["accuracy"] == 0.4 and m["confusion"]["supported->contradicted"] == 1
    none = l2.prf([("supported", "supported")])
    assert none["contradicted"]["precision"] is None and none["contradicted"]["recall"] is None
    assert l2.prf([])["accuracy"] is None


def test_verification_error_scores_as_refused_unverifiable(monkeypatch):
    monkeypatch.setattr(l2, "to_claim", lambda s, r: None)

    def boom(*_):
        raise IntegrityError("uncited numbers")

    monkeypatch.setattr(build, "verify_claim", boom)
    out = l2.predict(None, [{"id": "g1"}])
    assert out["g1"]["label"] == "unverifiable" and out["g1"]["refused"]
    assert out["g1"]["error"].startswith("IntegrityError")
    m, _ = l2.summarize([{"id": "g1", "gold": "contradicted"}], [], out)
    assert m["errors"] == 1 and m["contradiction_recall"] == 0.0

    def db_down(*_):
        raise OperationalError("select", {}, Exception("gone"))

    monkeypatch.setattr(build, "verify_claim", db_down)
    with pytest.raises(OperationalError):
        l2.predict(None, [{"id": "g1"}])


def test_headline_uses_real_set_only():
    real = [{"id": "g1", "gold": "supported"}, {"id": "g2", "gold": "supported"}]
    pert = [{"id": "p1", "gold": "contradicted"}]
    pred = {
        "g1": {"label": "contradicted"},
        "g2": {"label": "supported"},
        "p1": {"label": "contradicted"},
    }
    m, _ = l2.summarize(real, pert, pred)
    assert m["contradiction_precision"] == 0.0  # real set: one false contradiction
    assert m["combined_contradiction_precision"] == 0.5  # with the perturbation mixed in
    assert m["perturbed_contradiction_recall"] == 1.0


def test_ablation_patches_target_live_attributes():
    ev = EvidenceItem(
        agent="xbrl_table",
        stance="supports",
        summary="s",
        citations=[Citation(kind="xbrl_fact", ref_id=1, value=1.0, unit="USD")],
    )
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
    monkeypatch.setattr(
        l2,
        "predict",
        lambda s, recs, override=None, workers=1: {
            r["id"]: {"label": labels[r["id"]], "evidence": {}} for r in recs
        },
    )
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
    f = Filing(
        company_id=co.id,
        accession_no="acc-1",
        form_type="8-K",
        filed_at=date(2025, 7, 31),
        doc_url="u",
        storage_path="objects/x.htm",
        source_url="u",
        fetched_at=datetime.now(UTC),
        sha256="0" * 64,
        license_note="t",
    )
    db.add(f)
    db.flush()

    def add(a, b, claim_type, comparison=None, version=None):
        version = version or l2.extractor_version()
        c = Claim(
            company_id=co.id,
            filing_id=f.id,
            fiscal_year=2025,
            fiscal_period="Q3",
            span_start=a,
            span_end=b,
            text="t" * (b - a),
            claim_type=claim_type,
            comparison=comparison,
            metric="revenue",
            value=5.0,
            unit="percent",
            extractor_version=version,
        )
        db.add(c)
        db.flush()
        return c

    level = add(10, 30, "reported_result")  # exact span, wrong type
    add(5, 40, "comparison", "yoy")  # same type and comparison, IoU 20/35, older
    change = add(10, 45, "comparison", "yoy")  # same IoU, newer: wins the tie
    add(10, 45, "comparison", "yoy", version="s0-old")  # another extractor version: ignored
    add(100, 120, "comparison", "yoy")  # no overlap
    old = {"claim_type": "comparison", "comparison": "yoy"}
    recs = [
        {"id": "g1", "gold": "supported", "accession_no": "acc-1", "span": [10, 30], "claim": old},
        {
            "id": "g2",
            "gold": "supported",
            "accession_no": "acc-1",
            "span": [10, 30],
            "claim": {"claim_type": "reported_result", "comparison": None},
        },
        {
            "id": "g3",
            "gold": "unverifiable",
            "accession_no": "acc-1",
            "span": [60, 80],
            "claim": old,
        },
    ]
    out, unmatched = l2.remap_current(db, recs)
    assert unmatched == ["g3"]
    assert [(r["id"], r["source_claim_id"], r["span"]) for r in out] == [
        ("g1", change.id, [10, 45]),
        ("g2", level.id, [10, 30]),
    ]
    assert out[0]["gold"] == "supported" and out[0]["extractor_version"] == l2.extractor_version()
    assert out[0]["claim"]["value"] == 5.0  # a JSON float, as to_claim expects


def test_clopper_pearson_matches_exact_reference_values():
    # Reference values: beta quantiles (scipy.stats.beta.ppf) for the same k, n.
    assert l2.clopper_pearson(5, 10) == pytest.approx([0.187086, 0.812914], abs=1e-6)
    assert l2.clopper_pearson(0, 10) == pytest.approx([0.0, 0.308497], abs=1e-6)
    assert l2.clopper_pearson(10, 10) == pytest.approx([0.691503, 1.0], abs=1e-6)
    assert l2.clopper_pearson(3, 23) == pytest.approx([0.027752, 0.335889], abs=1e-6)
    assert l2.clopper_pearson(0, 0) is None


def test_summarize_reports_intervals_on_the_real_set():
    real = [
        {"id": "g1", "gold": "contradicted"},
        {"id": "g2", "gold": "supported"},
        {"id": "g3", "gold": "contradicted"},
    ]
    pred = {
        "g1": {"label": "contradicted"},
        "g2": {"label": "contradicted"},
        "g3": {"label": "supported"},
    }
    m, _ = l2.summarize(real, [], pred)
    assert m["contradiction_precision_ci"] == l2.clopper_pearson(1, 2)
    assert m["contradiction_recall_ci"] == l2.clopper_pearson(1, 2)
    assert m["accuracy_ci"] == l2.clopper_pearson(1, 3)
    lo, hi = m["accuracy_ci"]
    assert lo < m["accuracy"] < hi
    none, _ = l2.summarize([{"id": "g1", "gold": "supported"}], [], {"g1": {"label": "supported"}})
    assert none["contradiction_precision_ci"] is None and none["contradiction_recall_ci"] is None


def test_workers_are_validated_and_capped_below_the_pool():
    assert l2.cap_workers(8, pool_size=5) == 4  # the run's own Session holds one connection
    assert l2.cap_workers(2, pool_size=5) == 2
    assert l2.cap_workers(3, pool_size=1) == 1
    with pytest.raises(ValueError):
        l2.cap_workers(0, pool_size=5)
    with pytest.raises(SystemExit):
        l2.main(["--workers", "0", "--no-write"])


def test_heldout_set_scores_its_own_records_without_perturbations(monkeypatch):
    held = [{"id": "h001", "gold": "contradicted"}]
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return held

    monkeypatch.setattr(l2, "load", fake_load)
    monkeypatch.setattr(
        l2,
        "predict",
        lambda s, recs, override=None, workers=1: {
            r["id"]: {"label": "contradicted", "evidence": {}} for r in recs
        },
    )
    out = l2.run(set_name="heldout", ablate=False)
    assert loaded == [l2.HELDOUT]
    assert out["per_claim"] == {"h001": {"gold": "contradicted", "predicted": "contradicted"}}
    assert out["metrics"]["n_real"] == 1 and out["metrics"]["n_perturbed"] == 0

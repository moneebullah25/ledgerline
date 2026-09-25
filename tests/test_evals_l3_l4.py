import json

import pytest

from ledgerline import llm
from ledgerline.evals import l3, l4
from ledgerline.schemas import Citation, EvidenceItem


def test_cohen_kappa_reference_values():
    assert l3.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # Classic 2x2 example: po = 0.7, pe = 0.5 -> kappa 0.4
    a = ["y"] * 5 + ["n"] * 5
    b = ["y"] * 4 + ["n"] + ["y"] * 2 + ["n"] * 3
    assert l3.cohen_kappa(a, b) == pytest.approx(0.4)
    assert l3.cohen_kappa(["a", "a"], ["a", "a"]) is None  # pe == 1: undefined
    assert l3.cohen_kappa([], []) is None
    with pytest.raises(ValueError):
        l3.cohen_kappa(["a"], [])


def test_annotator_agreement_only_counts_second_pass_items():
    recs = [{"id": "g1", "pass1": "supported", "pass2": "supported"},
            {"id": "g2", "pass1": "supported", "pass2": "unverifiable"},
            {"id": "g3", "pass1": "contradicted", "pass2": None}]
    m = l3.annotator_agreement(recs)
    assert m["n_pass2"] == 2 and m["annotator_pass_agreement"] == 0.5
    assert m["pass2_disagreements"] == ["g2"]


def test_offline_run_does_not_report_a_judge_kappa(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    metrics, details = l3.run()
    assert metrics["judge_ran"] is False and metrics["judge_kappa"] is None
    assert metrics["n_pass2"] > 0 and "not_measured" in details


def test_judge_labels_each_claim_from_graph_evidence(monkeypatch):
    seen = []

    def fake(tier, system, data, instruction, schema):
        seen.append((tier, data["claim"]))
        return schema(label="contradicted", reason="r"), llm.Usage()

    monkeypatch.setattr(llm, "structured", fake)
    ev = EvidenceItem(agent="xbrl_table", stance="contradicts", summary="s",
                      citations=[Citation(kind="xbrl_fact", ref_id=1, value=1.0, unit="USD")])
    recs = [{"id": "g1", "claim": {"text": "Revenue was $1 billion"}}]
    out = l3.judge(recs, {"g1": {"evidence": {"xbrl_table": ev}}})
    assert out["g1"].label == "contradicted"
    assert seen == [("judge", "Revenue was $1 billion")]


def _row(ticker, release, flagged, missed, stance=None):
    return {"ticker": ticker, "release": release, "flagged": flagged, "missed": missed,
            "forecast_stance": stance or ("contradicts" if flagged else "supports")}


def test_backtest_summary_precision_base_rate_and_pairs():
    rows = [_row("A", "2025Q1", True, True), _row("A", "2025Q1", True, False),
            _row("B", "2025Q2", False, False), _row("B", "2025Q3", False, True),
            _row("C", "2025Q1", True, None)]  # outcome unknown: excluded
    m = l4.summarize(rows)
    assert m["backtest_precision"] == 0.5 and m["miss_rate_unflagged"] == 0.5
    assert m["n_outcome_known"] == 4 and m["n_flagged"] == 2 and m["pairs"] == 3
    assert m["flag_recall"] == 0.5 and m["pairs_target_met"] is False


def test_backtest_summary_with_no_flags_is_undefined_not_zero():
    m = l4.summarize([_row("A", "2025Q1", False, False, "insufficient")])
    assert m["backtest_precision"] is None and m["n_forecast_decided"] == 0


def test_judge_records_a_bad_output_per_claim_instead_of_aborting(monkeypatch):
    def fake(tier, system, data, instruction, schema):
        if data["claim"] == "bad":
            raise ValueError("judge: output failed validation")
        return schema(label="supported", reason="r"), llm.Usage()

    monkeypatch.setattr(llm, "structured", fake)
    recs = [{"id": "g1", "claim": {"text": "bad"}}, {"id": "g2", "claim": {"text": "ok"}}]
    out = l3.judge(recs, {"g1": {"evidence": {}}, "g2": {"evidence": {}}})
    assert out["g1"].startswith("ValueError") and out["g2"].label == "supported"

    def limited(*_):
        raise llm.RateLimited("judge: rate-limited")
    monkeypatch.setattr(llm, "structured", limited)
    with pytest.raises(llm.RateLimited):
        l3.judge(recs[1:], {"g2": {"evidence": {}}})
    with pytest.raises(ValueError):
        l3.judge(recs, {}, workers=0)


def test_run_skips_and_counts_claims_without_evidence(monkeypatch, tmp_path):
    recs = [{"id": "g1", "gold": "supported", "claim": {"text": "a"}},
            {"id": "g2", "gold": "contradicted", "claim": {"text": "b"}},
            {"id": "g3", "gold": "supported", "claim": {"text": "c"}}]
    monkeypatch.setattr(l3, "load", lambda path: recs)
    monkeypatch.setattr(llm, "available", lambda: True)
    labels = {"a": "supported", "b": "contradicted"}

    def fake(tier, system, data, instruction, schema):
        if data["claim"] not in labels:
            raise ValueError("bad output")
        return schema(label=labels[data["claim"]], reason="r"), llm.Usage()
    monkeypatch.setattr(llm, "structured", fake)
    ev = {"agent": "xbrl_table", "stance": "insufficient", "summary": "s"}
    # g1, g2 have evidence; g3 errored in L2 (empty evidence); an unmatched id is simply absent.
    path = tmp_path / "ev.json"
    path.write_text(json.dumps({"g1": {"xbrl_table": ev}, "g2": {"xbrl_table": ev}, "g3": {}}))
    metrics, details = l3.run(str(path))
    assert metrics["judge_n"] == 2 and metrics["judge_skipped_no_evidence"] == 1
    assert metrics["judge_errors"] == 0 and metrics["judge_kappa"] == 1.0
    assert details["skipped_no_evidence"] == ["g3"]

    labels.pop("b")  # g2's judge output now fails validation: recorded, not fatal
    metrics, details = l3.run(str(path))
    assert metrics["judge_n"] == 1 and metrics["judge_errors"] == 1
    assert list(details["judge_errors"]) == ["g2"]

import json

import pytest

from ledgerline import llm
from ledgerline.evals import golden, l3, l4


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


def test_main_exits_1_without_llm(monkeypatch, capsys):
    monkeypatch.setattr(llm, "available", lambda: False)
    with pytest.raises(SystemExit) as e:
        l3.main(["--no-write"])
    assert e.value.code == 1 and l3.NO_LLM in capsys.readouterr().err
    with pytest.raises(RuntimeError):
        l3.run()


def test_kappa_ci_and_confusion_from_known_matrix():
    # gold x pred: S->S 4, S->U 1, C->C 3, U->U 2 ; po = 0.9
    gold = ["supported"] * 5 + ["contradicted"] * 3 + ["unverifiable"] * 2
    pred = ["supported"] * 4 + ["unverifiable"] + ["contradicted"] * 3 + ["unverifiable"] * 2
    # pe = (5*4 + 3*3 + 2*3) / 100 = 0.35 -> kappa = 0.55 / 0.65
    assert l3.cohen_kappa(gold, pred) == pytest.approx(0.55 / 0.65)
    m = l3.confusion(gold, pred)
    assert m["supported"] == {"supported": 4, "contradicted": 0, "unverifiable": 1}
    assert m["unverifiable"]["unverifiable"] == 2
    lo, hi = l3.bootstrap_ci(gold, pred)
    assert lo <= l3.cohen_kappa(gold, pred) <= hi <= 1.0
    assert l3.bootstrap_ci(["a"], ["a"]) is None


def test_cli_args():
    a = l3.parse_args([])
    assert (a.set_name, a.workers, a.evidence, a.no_write) == ("golden", 1, None, False)
    a = l3.parse_args(["--set", "heldout", "--workers", "4", "--evidence", "e.json"])
    assert (a.set_name, a.workers, a.evidence) == ("heldout", 4, "e.json")
    for bad in (["--set", "nope"], ["--workers", "0"]):
        with pytest.raises(SystemExit):
            l3.parse_args(bad)


def test_run_judges_only_needs_judgment_claims(monkeypatch, tmp_path):
    claim = {k: None for k in golden.FIELDS} | {"claim_type": "reported_result"}
    recs = [{"id": f"g{i}", "gold": g, "claim": claim | {"text": t}}
            for i, (g, t) in enumerate([("supported", "a"), ("contradicted", "b"),
                                        ("supported", "c"), ("supported", "d")])]
    monkeypatch.setattr(l3, "load", lambda path: recs)
    monkeypatch.setattr(llm, "available", lambda: True)
    cite = [{"kind": "xbrl_fact", "ref_id": 1, "value": 1.0, "unit": "USD"}]
    rag = {"agent": "filing_rag", "stance": "supports", "summary": "s", "citations": cite}
    xbrl = {"agent": "xbrl_table", "stance": "supports", "summary": "s", "citations": cite}
    path = tmp_path / "ev.json"  # g0, g1 need judgment; g2 is rule-decided; g3 has no evidence
    path.write_text(json.dumps({"g0": {"filing_rag": rag}, "g1": {"filing_rag": rag},
                                "g2": {"xbrl_table": xbrl}, "g3": {}}))
    monkeypatch.setattr(l3.critic, "critique", lambda c, e: (None, llm.Usage()))

    def fake(c, e, crit):
        if c.text == "b":
            raise ValueError("bad output")
        return type("V", (), {"label": "supported"}), llm.Usage(), "claude-opus-5"
    monkeypatch.setattr(l3.adjudicator, "adjudicate", fake)
    m, d = l3.run(str(path))
    assert (m["n_total"], m["judge_n"], m["skipped_deterministic"], m["skipped_no_evidence"],
            m["judge_errors"], m["adjudicator_fallbacks"]) == (4, 1, 1, 1, 1, 0)
    assert list(d["judge_errors"]) == ["g1"] and d["skipped_no_evidence"] == ["g3"]


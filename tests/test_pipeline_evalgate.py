from ledgerline.evals.gate import compare, render, verdict_diff


def run(p: float, **per_claim) -> dict:
    return {"metrics": {"contradiction_precision": p},
            "per_claim": {k: {"gold": "contradicted", "predicted": v} for k, v in per_claim.items()}}


def test_absolute_two_point_threshold():
    assert compare(run(0.70), run(0.72))[0]          # exactly 2 pts: allowed
    assert not compare(run(0.699), run(0.72))[0]     # 2.1 pts: blocked
    assert compare(run(0.80), run(0.72))[0]          # improvement
    assert compare(run(0.10), {"metrics": {"contradiction_precision": None}})[0]


def test_verdict_diff_lists_only_changes():
    base = run(0.7, a="contradicted", b="supported", gone="supported")
    cur = run(0.7, a="contradicted", b="contradicted", new="unverifiable")
    diff = verdict_diff(cur, base)
    assert diff == [("b", "supported", "contradicted", "contradicted"),
                    ("gone", "supported", None, "contradicted"),
                    ("new", None, "unverifiable", "contradicted")]
    assert "| b | supported | contradicted |" in render(True, "ok", diff)


def test_missing_metric_fails():
    assert not compare({"metrics": {}}, run(0.72))[0]

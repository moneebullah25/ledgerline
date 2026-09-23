from ledgerline.report.severity import load_weights, severity

W = load_weights()


def test_ordering():
    contradicted_confident_unhedged = severity("contradicted", 0.9, 0.08, 0.0, 0.03, W)
    contradicted_hedged = severity("contradicted", 0.9, 0.08, 0.9, 0.03, W)
    contradicted_unsure_small = severity("contradicted", 0.5, 0.01, 0.0, None, W)
    unverifiable_hedged = severity("unverifiable", 0.6, None, 0.8, None, W)
    supported = severity("supported", 0.99, 0.0, 0.0, 0.05, W)
    assert (contradicted_confident_unhedged > contradicted_hedged
            > contradicted_unsure_small > unverifiable_hedged > supported == 0)


def test_caps_bound_outliers():
    # A 400% "deviation" (unit mix-up upstream) must not dwarf everything else.
    assert severity("contradicted", 0.9, 4.0, 0.0, 2.0, W) == severity(
        "contradicted", 0.9, W.deviation_cap, 0.0, W.abnormal_return_cap, W)

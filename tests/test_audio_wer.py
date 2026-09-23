from ledgerline.evals.wer import _tokens, align, label_accuracy, normalize


def test_normalize_matches_spoken_and_written_forms():
    assert normalize("Inflation fell to 2.2 percent. [Laughter]") == normalize(
        "inflation fell to 2.2%")


def test_align_cuts_reference_at_excerpt_end():
    ref = ["a", "b", "c", "d", "e", "f", "g", "h"]
    cut, pairs = align(ref, ["a", "b", "x", "d", "e"])
    assert cut == 5  # ref beyond "e" is outside the excerpt
    assert pairs == {0: 0, 1: 1, 3: 3, 4: 4}


def test_align_ignores_stray_short_match_far_ahead():
    ref = [f"w{i}" for i in range(20)] + ["thank", "you"]
    hyp = [f"w{i}" for i in range(8)] + ["thank", "you"]
    cut, pairs = align(ref, hyp)
    assert cut == 8
    assert set(pairs) == set(range(8))


def test_label_accuracy_by_aligned_words():
    ref_turns = [{"text": "good afternoon everyone", "role": "executive"},
                 {"text": "steve", "role": "operator"},
                 {"text": "thank you mr chairman", "role": "analyst"}]
    hyp_turns = [{"text": "good afternoon everyone steve", "role": "executive"},
                 {"text": "thank you mister chairman", "role": "executive"}]
    ref, ref_owner = _tokens(ref_turns)
    hyp, hyp_owner = _tokens(hyp_turns)
    _, pairs = align(ref, hyp)
    acc = label_accuracy(hyp_turns, hyp_owner, ref_turns, ref_owner, pairs, "role")
    assert acc["turns"] == 2 and acc["turn_acc"] == 0.5
    assert acc["word_acc"] == 3 / 8  # 3 of 4 words right in turn 0, 0 of 4 in turn 1

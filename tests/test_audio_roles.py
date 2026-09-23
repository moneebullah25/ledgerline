from ledgerline.audio.roles import assign_roles, qa_start, sections, seed_roles
from ledgerline.ingest.fomc import parse_transcript

CALL = [
    ("Good day, ladies and gentlemen, and welcome to the Q3 earnings call. After the speakers' "
     "remarks there will be a question-and-answer session."),
    "Thank you. I'll now turn the call over to Jane Doe, our CFO.",
    "Thanks. This quarter we reported revenue of $94 billion.",
    ("We will now begin the question-and-answer session. Our first question comes from "
     "John Smith with Big Bank."),
    "Hi, thanks for taking my question. Can you talk about margins?",
    "Sure. Margins were 46%.",
]
SPK = ["spk0", "spk1", "spk1", "spk0", "spk2", "spk1"]


def test_seeds_this_and_next_speaker():
    own, prev = seed_roles(CALL)
    assert own[0] == ["operator"]
    assert prev[2] == ["executive"] and own[2] == ["executive"]  # handed over + "we reported"
    assert own[3] == ["operator"]
    assert own[4] == ["analyst"] and prev[4] == ["analyst"]


def test_majority_vote_per_cluster():
    # turn 5 has no cue of its own; its cluster (spk1) votes executive.
    assert assign_roles(CALL, SPK) == [
        "operator", "executive", "executive", "operator", "analyst", "executive"]
    assert assign_roles(["Hello there."], ["spk9"]) == ["unknown"]


def test_own_cue_beats_cluster_and_one_stray_turn_cannot_flip_it():
    texts = ["Good afternoon. My colleagues and I decided. I look forward to your questions.",
             "Michael McKee with Bloomberg. What would it take?",
             "Well, it depends.", "Steve.", "Your other question was?"]
    # A reporter and a next-speaker cue landed in the chair's cluster "a".
    assert assign_roles(texts, ["a", "a", "a", "b", "a"]) == [
        "executive", "analyst", "executive", "operator", "executive"]


def test_handing_back_to_operator_does_not_seed_executive():
    assert seed_roles(["I'll turn it back over to the operator.", "Next caller."])[1][1] == []


def test_section_split_skips_early_qa_mention():
    assert qa_start(CALL) == 4
    assert sections(CALL) == ["prepared_remarks"] * 4 + ["qa"] * 2
    assert qa_start(["Just remarks."]) is None


def test_fomc_split_and_roles():
    texts = ["Good afternoon. My colleagues and I remain focused. I look forward to your questions.",
             "Steve.", "Steve Liesman, CNBC. Thank you, Mr. Chairman.", "So, a lot in there."]
    assert seed_roles(["Howard Schneider with Reuters. How close was it?"])[0] == [["analyst"]]
    assert sections(texts) == ["prepared_remarks", "qa", "qa", "qa"]
    assert assign_roles(texts, ["a", "b", "c", "a"]) == ["executive", "operator", "analyst",
                                                         "executive"]


def test_parse_fomc_transcript():
    text = ("September 18, 2024\n\nChair Powell’s Press Conference\n\nFINAL\n\n"
            "Transcript of Chair Powell’s Press Conference\nSeptember 18, 2024\n"
            "CHAIR POWELL. Good afternoon. Rates fell by\nhalf a point.\nPage 1 of 26\n"
            "MICHELLE SMITH. Steve.\nSTEVE LIESMAN. Steve Liesman, CNBC. Why?\n"
            "CHAIR POWELL. Because.\nMICHELLE SMITH. Thank you.\n")
    turns = parse_transcript(text)
    assert [(t["speaker"], t["role"]) for t in turns] == [
        ("CHAIR POWELL", "executive"), ("MICHELLE SMITH", "operator"),
        ("STEVE LIESMAN", "analyst"), ("CHAIR POWELL", "executive"),
        ("MICHELLE SMITH", "operator")]
    assert turns[0]["text"] == "Good afternoon. Rates fell by half a point."

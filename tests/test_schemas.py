import pytest
from pydantic import ValidationError

from ledgerline.schemas import Citation, ClaimDraft, VerdictDraft, uncited_numbers


def test_uncited_number_is_flagged():
    cites = [Citation(kind="xbrl_fact", ref_id=1, value=94_930_000_000)]
    assert uncited_numbers("Revenue was $94.9 billion.", cites) == []
    assert uncited_numbers("Revenue was $97.2 billion.", cites) == [97.2]


def test_claim_numbers_and_small_ordinals_allowed():
    assert uncited_numbers("In Q3 2024 the 4 agents saw 12.5%", [], "grew 12.5%") == []


def test_verdict_must_cite_unless_unverifiable():
    with pytest.raises(ValidationError):
        VerdictDraft(label="supported", confidence=0.9, rationale="x", citations=[])
    VerdictDraft(label="unverifiable", confidence=0.0, rationale="no evidence", citations=[])


def test_claim_draft_rejects_value_without_unit():
    with pytest.raises(ValidationError):
        ClaimDraft(text="rev 5", span_start=0, span_end=5, claim_type="reported_result", value=5)


@pytest.mark.parametrize(
    "text",
    [
        "Revenue was $25 billion.",
        "Capex was $2,050 million.",
        "Sales grew 12%.",
        "Margin rose 3 points.",
        "Revenue reached 2050 million USD.",
        "A $ 7 charge.",
        "Revenue was €25 billion.",
        "Revenue was £25 billion.",
        "Revenue was USD 25 billion.",
        "A 25-billion buyback.",
        "Revenue of 25  billion.",
        "Revenue of 25\nbillion.",
    ],
)
def test_amounts_are_not_exempt_as_ordinals_or_years(text):
    assert uncited_numbers(text, []) != []


def test_bare_ordinals_and_years_stay_exempt():
    assert uncited_numbers("In fiscal 2025 Q3, 4 agents ran over 12 months.", []) == []
    assert uncited_numbers("Per the 10-K and the 8-K.", []) == []


def test_negative_cited_value_may_print_as_magnitude():
    loss = [Citation(kind="xbrl_fact", ref_id=1, value=-2_100_000_000.0, unit="USD")]
    assert uncited_numbers("Net loss of $2.1 billion.", loss) == []
    assert uncited_numbers("Net income of -2.10 billion USD.", loss) == []
    assert uncited_numbers("Net loss of $2.4 billion.", loss) == [2.4]

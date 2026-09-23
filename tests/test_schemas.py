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

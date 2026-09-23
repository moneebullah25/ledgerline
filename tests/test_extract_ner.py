import pytest

from ledgerline.extract.pipeline import extract_claims

pytestmark = pytest.mark.ml


def test_gliner_spans_and_segment_qualifier():
    from ledgerline.extract.ner import entity_spans

    text = "Data Center revenue was $89.0 billion, up 117% from a year ago."
    spans = {(text[s:e], label) for s, e, label in entity_spans(text)}
    assert ("$89.0 billion", "money") in spans and ("117%", "percent") in spans
    assert any(label == "metric" and "revenue" in t for t, label in spans)
    (c, *_) = extract_claims(text, fiscal_year=2027, fiscal_period="Q2")
    assert c.metric is None  # a segment figure is never company-wide revenue

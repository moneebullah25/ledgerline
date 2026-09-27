import json
import re

import pytest

from ledgerline import llm
from ledgerline.extract import refine
from ledgerline.extract.pipeline import extract_claims
from ledgerline.extract.refine import Normalized, apply
from ledgerline.schemas import ClaimDraft

TEXT = ("Alphabet Announces First Quarter 2026 Results\n"
        "Operating income was $39.7 billion. "
        "Google Cloud saw a meaningful acceleration in growth as revenues increased 63% to $20.0 billion.\n")


def ok(**kw) -> Normalized:
    base = {"is_company_total": True, "metric": "revenue", "target_fiscal_year": 2026, "target_fiscal_period": "Q1",
                "is_gaap": True, "value": 20e9, "unit": "USD", "reason": ""}
    return Normalized(**base | kw)


def echo(instruction: str, **kw) -> Normalized:
    """Confirms the rules' reading (quoted in the instruction) except for the fields in kw."""
    reading = json.loads(re.search(r"as: (\{.*?\})\.", instruction)[1])
    value = abs(reading["value"]) if reading["value"] is not None else None
    return ok(metric=reading["metric"], value=value, unit=reading["unit"],
              target_fiscal_year=reading["target_fiscal_year"],
              target_fiscal_period=reading["target_fiscal_period"]).model_copy(update=kw)


def draft(**kw) -> ClaimDraft:
    base = {"text": "revenues were $20.0 billion", "span_start": 0, "span_end": 10, "claim_type": "reported_result",
                "metric": "revenue", "value": 20e9, "unit": "USD", "target_fiscal_year": 2026, "target_fiscal_period": "Q1"}
    return ClaimDraft(**base | kw)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """A live-looking llm whose answer comes from `answer(data, instruction)`; calls are recorded."""
    calls = []
    monkeypatch.setattr(refine, "_cache_path", lambda: tmp_path / "cache.json")
    monkeypatch.setattr(refine, "_cache", None)

    def install(answer, usage=llm.OFFLINE):
        monkeypatch.setattr(llm, "available", lambda: True)
        def structured(tier, system, data, instruction, schema, max_tokens=4000):
            calls.append(data)
            return answer(data, instruction), usage
        monkeypatch.setattr(llm, "structured", structured)
        return calls
    return install


def run(text=TEXT, **kw):
    return extract_claims(text, fiscal_year=2026, fiscal_period="Q1", use_ner=False, **kw)


def test_offline_is_a_no_op():
    assert not llm.available()
    d = draft()
    assert refine.refine(TEXT, [d], 2026, "Q1", None) == ([d], llm.OFFLINE)


def test_segment_revenue_loses_its_metric_but_keeps_its_value(stub):
    before = {(c.metric, c.value) for c in run()}
    assert ("revenue", 20e9) in before and ("revenue", 63.0) in before  # the rules read Cloud as the company
    stub(lambda data, ins: echo(ins, is_company_total="Cloud" not in data["sentence"]))
    got = {(c.metric, c.claim_type, c.value) for c in run()}
    assert ("operating_income", "reported_result", 39.7e9) in got
    assert (None, "qualitative", 20e9) in got and (None, "qualitative", 63.0) in got


def test_context_blocks(stub):
    calls = stub(lambda data, ins: echo(ins))
    run()
    c = next(d for d in calls if "20.0" in d["claim"])
    assert c["sentence"].startswith("Google Cloud saw") and c["claim"] != c["sentence"]
    assert c["sentence_before"] == "Operating income was $39.7 billion."
    assert c["section_heading"] == "Alphabet Announces First Quarter 2026 Results"


def test_table_row_context_carries_the_header_rows(stub):
    text = ("Intro paragraph.\nFull-Year 2025 Financial Results\n(In millions)\nGAAP | Non-GAAP\n2025 | 2024 | vs. 2024\n"
            "Revenue | 52,900 | 53,100 | flat\nCost of sales:\nProducts | 1 | 2 | up 1 ppt\n"
            "Gross margin | 34.8% | 32.7% | up 2.1 ppts\nTax rate | 9% | 8% | up 1 ppt\n")
    d = draft(text="Gross margin | 34.8%", span_start=text.index("Gross"), span_end=text.index("Gross") + 20,
              metric="gross_margin", value=34.8, unit="percent", target_fiscal_year=2025, target_fiscal_period="Q4")
    ctx = refine.context(text, d, 2025, "Q4", None)
    assert ctx["sentence_before"] == ("Full-Year 2025 Financial Results\n(In millions)\n"
                                      "GAAP | Non-GAAP\n2025 | 2024 | vs. 2024")
    assert ctx["sentence_after"].startswith("Tax rate")
    assert ctx["section_heading"] == "(In millions)"  # the nearest heading-like line


def test_table_row_context_stops_at_the_previous_table():
    text = ("Old | 2024\nCash | 5 | 6\nSummary\nQ1 2026 | Q1 2025\nRevenue | 52,900 | 53,100\n")
    d = draft(text="Revenue | 52,900", span_start=text.index("Revenue"), span_end=text.index("Revenue") + 16)
    assert refine.context(text, d, 2026, "Q1", None)["sentence_before"] == "Summary\nQ1 2026 | Q1 2025"


def test_comparison_keeps_its_concepts_unit(stub):
    """Blocker: a margin change re-mapped to gross_profit growth would be scored as a dollar fact."""
    stub(lambda data, ins: echo(ins, metric="gross_profit"))
    got = {(c.claim_type, c.metric, c.value) for c in run("Gross margin expanded 120 basis points.")}
    assert got == {("qualitative", None, 1.2)}


def test_refine_after_dedupe_and_roles(stub):
    calls = stub(lambda data, ins: echo(ins))
    exec_ = "Revenue was $20.0 billion. "
    text = exec_ + "Revenue was $20.0 billion. " + "Did revenue reach $30.0 billion?"
    roles = [(0, 27, "executive"), (27, 54, "executive"), (54, len(text), "analyst")]
    got = run(text, speaker_roles=roles)
    assert [(c.value, c.speaker_role) for c in got] == [(20e9, "executive")]
    assert len(calls) == 1  # no call for the restatement or the analyst's number


def test_usage_total_and_progress_batches(stub, monkeypatch):
    one = llm.Usage(model="claude-haiku-4-5", input_tokens=10, cache_write_tokens=2)
    stub(lambda data, ins: echo(ins), usage=one)
    monkeypatch.setattr(refine, "BATCH", 2)
    drafts = [draft(text=f"span {i}", span_start=i, span_end=i + 1) for i in range(5)]
    seen = []
    out, total = refine.refine("x" * 10, drafts, 2026, "Q1", None, seen.append)
    assert len(out) == 5
    assert (total.model, total.input_tokens, total.cache_write_tokens) == ("claude-haiku-4-5", 50, 10)
    assert [u.input_tokens for u in seen] == [20, 20, 10]  # every BATCH answers, then the rest
    _, cached = refine.refine("x" * 10, drafts, 2026, "Q1", None)
    assert cached == llm.OFFLINE  # cache hits cost nothing


def test_cache_survives_corruption_and_schema_drift(stub, tmp_path):
    calls = stub(lambda data, ins: echo(ins))
    (tmp_path / "cache.json").write_text("{not json")
    first = run()
    n = len(calls)
    assert n and json.loads((tmp_path / "cache.json").read_text())  # rewritten, valid
    assert not list(tmp_path.glob("*.tmp"))
    refine._cache = {k: {"metric": "no-such-metric"} for k in refine._cache}  # no longer validates
    assert run() == first and len(calls) == 2 * n  # re-asked, not crashed


def test_cache_key_includes_the_model(stub, monkeypatch):
    calls = stub(lambda data, ins: echo(ins))
    run()
    n = len(calls)
    monkeypatch.setitem(llm.TIERS, "fast", "claude-haiku-9")
    run()
    assert len(calls) == 2 * n


def test_unreachable_model_fails_the_document_at_once(stub):
    def down(data, instruction):
        raise llm.LLMUnavailable("down")
    calls = stub(down)
    with pytest.raises(llm.LLMUnavailable):
        refine.refine("x" * 50, [draft(text=f"s{i}", span_start=i, span_end=i + 1) for i in range(40)], 2026, "Q1", None)
    assert len(calls) < 40  # not retried per claim; queued calls are cancelled


def test_answers_are_cached_and_a_failed_call_fails_the_document(stub):
    calls = stub(lambda data, ins: echo(ins, is_company_total=False))
    first = run()
    n = len(calls)
    assert n and run() == first and len(calls) == n  # the re-run is served from the cache
    refine._cache = None  # a fresh process reads the file
    assert run() == first and len(calls) == n

    def boom(data, instruction):
        raise ValueError("refused")
    calls = stub(boom)
    with pytest.raises(ValueError, match="failed twice"):  # the document fails, a re-run retries
        refine.refine("revenues were $20.0 billion", [draft(text="a new span")], 2026, "Q1", None)
    assert len(calls) == n + 2


@pytest.mark.parametrize("d,r,want", [
    # full-year figure labelled Q4 -> FY
    (draft(target_fiscal_year=2025, target_fiscal_period="Q4", metric="gross_margin", value=34.8, unit="percent"),
     ok(metric="gross_margin", value=34.8, unit="percent", target_fiscal_year=2025, target_fiscal_period="FY"),
     ("gross_margin", 2025, "FY")),
    # dividends read as operating cash flow: remapped to a concept of another unit -> None
    (draft(metric="operating_cash_flow"), ok(metric="eps_diluted"), (None, 2026, "Q1")),
    # re-map within the same unit is allowed
    (draft(metric="gross_profit"), ok(metric="revenue"), ("revenue", 2026, "Q1")),
    # the model reads the number at another scale: metric dropped, value untouched
    (draft(), ok(value=20e6), (None, 2026, "Q1")),
    (draft(value=9.2), ok(value=9.2e9), (None, 2026, "Q1")),
    # ... unless it is a table row, whose scale the rules took from the header
    (draft(text="Gross margin | 54,781", metric="gross_profit"), ok(metric="gross_profit", value=54781e3),
     ("gross_profit", 2026, "Q1")),
    # another number altogether (the level next to a change) is not a scale correction
    (draft(claim_type="comparison", comparison="yoy", metric="eps_diluted", value=29.0, unit="percent"),
     ok(metric="eps_diluted", value=2.02, unit="USD/share"), ("eps_diluted", 2026, "Q1")),
    # a loss is stored negative, the model reads it positive
    (draft(metric="net_income", value=-1.2e9), ok(metric="net_income", value=1.2e9), ("net_income", 2026, "Q1")),
    # a range matches on either bound
    (draft(value=19e9, value_high=20e9), ok(), ("revenue", 2026, "Q1")),
    # non-GAAP
    (draft(), ok(is_gaap=False), (None, 2026, "Q1")),
    # an actual cannot move into the future, nor years away
    (draft(), ok(target_fiscal_period="Q3"), ("revenue", 2026, "Q1")),
    (draft(), ok(target_fiscal_year=2019), ("revenue", 2026, "Q1")),
    # fiscal-year offset: the baseline year was taken as the target
    (draft(target_fiscal_year=2025), ok(), ("revenue", 2026, "Q1")),
    # guidance may point ahead; a comparison's percent can carry any metric
    (draft(claim_type="guidance", target_fiscal_period="Q2"), ok(target_fiscal_period="Q3"), ("revenue", 2026, "Q3")),
    (draft(claim_type="comparison", comparison="yoy", metric="revenue", value=6.6, unit="percent"),
     ok(metric="net_income", value=6.6, unit="percent"), ("net_income", 2026, "Q1")),
    # no fiscal period (trailing twelve months)
    (draft(), ok(target_fiscal_year=None, target_fiscal_period=None), ("revenue", None, None)),
    # a claim the rules gave no period never gains one
    (draft(claim_type="guidance", target_fiscal_year=None, target_fiscal_period=None), ok(), ("revenue", None, None)),
    # an unmapped claim never gains a metric
    (draft(metric=None), ok(), (None, 2026, "Q1")),
    # a period without a year is incomplete: the rules' period stays (was a TypeError)
    (draft(), ok(target_fiscal_year=None, target_fiscal_period="Q4"), ("revenue", 2026, "Q1")),
])
def test_apply_guards(d, r, want):
    got = apply(d, r, 2026, "Q1")
    assert (got.metric, got.target_fiscal_year, got.target_fiscal_period) == want
    assert (got.value, got.value_high, got.text) == (d.value, d.value_high, d.text)


def test_heading_period_is_not_overridden():
    # MCD Q4 2024 (held-out h054): the model moved a bullet under "Fourth quarter financial
    # performance:" to FY; the heading names Q4 and the sentence names no period
    d = draft(claim_type="comparison", comparison="yoy", metric="operating_income", value=2.0,
              unit="percent", target_fiscal_year=2024, target_fiscal_period="Q4")
    r = ok(metric="operating_income", value=2.0, unit="percent", target_fiscal_year=2024,
           target_fiscal_period="FY")
    blocks = {"sentence": "Consolidated operating income increased 2% (3% in constant currencies).",
              "section_heading": "Fourth quarter financial performance:"}
    assert apply(d, r, 2024, "Q4", blocks, "1231").target_fiscal_period == "Q4"
    # a sentence that names its own period may still be corrected; so may an undated heading
    dated = blocks | {"sentence": "Full year operating income increased 2%."}
    assert apply(d, r, 2024, "Q4", dated, "1231").target_fiscal_period == "FY"
    assert apply(d, r, 2024, "Q4", blocks | {"section_heading": ""}, "1231").target_fiscal_period == "FY"

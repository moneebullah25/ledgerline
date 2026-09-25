from datetime import date, timedelta

import numpy as np
import pytest

from ledgerline.graph.agents import filing_rag, forecast, market_reaction
from ledgerline.models import Claim
from ledgerline.store.retrieval import Hit
from ledgerline.store.xbrl_query import Fact


def C(**kw) -> Claim:
    base = {"company_id": 1, "fiscal_year": 2025, "fiscal_period": "Q1", "span_start": 0,
            "span_end": 1, "claim_type": "reported_result", "filing_id": 1}
    return Claim(**(base | kw))


# ---- market reaction

def test_abnormal_return_window():
    d0 = date(2025, 1, 30)  # Thursday
    days = [d0 + timedelta(days=i) for i in (-3, -2, -1, 0, 1, 4)]
    aapl = [100, 100, 101, 104, 110, 111]
    spy = [500, 500, 500, 505, 510, 512]
    prices = [(i, "AAPL", d, p) for i, (d, p) in enumerate(zip(days, aapl))]
    prices += [(100 + i, "SPY", d, p) for i, (d, p) in enumerate(zip(days, spy))]
    ev = market_reaction.assess("AAPL", d0, prices)
    # day 0 = Jan 30, window close(-2)=Jan 28 -> close(+1)=Jan 31
    assert ev.payload["abnormal_return"] == pytest.approx(110 / 100 - 510 / 500)
    assert ev.stance == "insufficient" and {c.kind for c in ev.citations} == {"price"}
    assert market_reaction.assess("AAPL", d0 + timedelta(days=30), prices).citations == []


# ---- filing RAG

HIT = Hit(1, 9, "Part I > Item 2. MD&A", "Total net sales were $94,930 million. "
          "Services grew strongly. iPhone sales declined.", 1.0, 0, 90)


def fake_scores(entail_on: str):
    """contradiction/entailment/neutral probs: entail windows containing `entail_on`."""
    return lambda pairs: np.array([[0.05, 0.9, 0.05] if entail_on in w else [0.1, 0.1, 0.8]
                                   for w, _ in pairs])


def test_windows_are_exact_substrings():
    ws = filing_rag.windows(HIT.text)
    assert len(ws) == 2 and all(w in HIT.text for w in ws)


def test_numeric_claim_needs_the_number_in_the_quote():
    claim = C(text="revenue of $94.9 billion", metric="revenue", value=94.9e9, unit="USD")
    ev = filing_rag.assess(claim, [HIT], fake_scores("94,930"))
    assert ev.stance == "supports" and "94,930" in ev.citations[0].quote
    # The model "entails" a window without the number: must not count.
    ev = filing_rag.assess(claim, [HIT], fake_scores("iPhone"))
    assert ev.stance == "insufficient"
    other = C(text="revenue of $120 billion", metric="revenue", value=120e9, unit="USD")
    assert filing_rag.assess(other, [HIT], fake_scores("")).stance == "insufficient"


def test_offline_nli_never_contradicts():
    claim = C(text="Services grew strongly", claim_type="qualitative")
    contra = lambda pairs: np.array([[0.99, 0.005, 0.005]] * len(pairs))
    ev = filing_rag.assess(claim, [HIT], contra)
    assert ev.stance == "insufficient" and ev.payload["p_contra"] == pytest.approx(0.99)


def test_qualitative_claim_supported_by_nli():
    claim = C(text="Services grew strongly", claim_type="qualitative")
    ev = filing_rag.assess(claim, [HIT], fake_scores("Services"))
    assert ev.stance == "supports" and ev.payload["margin"] == pytest.approx(0.85)


# ---- forecast

def test_mase_is_relative_to_seasonal_naive():
    actual = np.array([10.0, 12, 14, 16])
    naive = np.array([8.0, 10, 12, 14])  # MAE 2
    assert forecast.mase(np.array([9.0, 11, 13, 15]), naive, actual) == pytest.approx(0.5)
    assert forecast.mase(naive, naive, actual) == pytest.approx(1.0)


def _fact(fy, fp, v):
    return Fact(v, "USD", (1,), "revenue", fy, fp, date(fy, 1, 1), False, "u")


def test_contiguous_tail_drops_history_before_a_gap():
    fs = [_fact(2023, "Q3", 1), _fact(2024, "Q1", 2), _fact(2024, "Q2", 3), _fact(2024, "Q3", 4),
          _fact(2024, "Q4", 5), _fact(2025, "Q1", 6)]
    assert [f.value for f in forecast.contiguous_tail(fs)] == [2, 3, 4, 5, 6]


def test_short_history_is_insufficient():
    claim = C(text="revenue of $10", claim_type="guidance", metric="revenue", value=10.0,
              unit="USD", target_fiscal_year=2025, target_fiscal_period="Q2")
    ev = forecast.assess(claim, [_fact(2025, "Q1", 9.0)])
    assert ev.stance == "insufficient" and "too short" in ev.summary


@pytest.mark.ml
def test_forecast_on_seasonal_series():
    rng = np.random.default_rng(0)
    quarters = [(2015 + i // 4, f"Q{i % 4 + 1}") for i in range(40)]
    y = [100 + 2 * i + 10 * (i % 4 == 3) + rng.normal(0, 1) for i in range(40)]
    hist = [_fact(fy, fp, v) for (fy, fp), v in zip(quarters, y)]  # ends 2024 Q4
    fy, fp = 2024, "Q4"
    guided = 100 + 2 * 40  # the trend's next value, 2025 Q1
    claim = C(text=f"revenue of ${guided}", claim_type="guidance", metric="revenue",
              value=float(guided), unit="USD", fiscal_year=fy, fiscal_period=fp,
              target_fiscal_year=2025, target_fiscal_period="Q1")
    ev = forecast.assess(claim, hist)
    assert ev.payload["mase"]["ensemble"] < 1.0
    assert ev.stance == "supports"
    claim.value = 400.0
    assert forecast.assess(claim, hist).stance == "contradicts"


# ---- filing RAG, online judgment with a stubbed llm.structured

def _judge(monkeypatch, **j):
    from ledgerline import llm
    out = filing_rag._Judgment(**({"stance": "supports", "passage": 0,
                                   "quote": "Services grew strongly.", "summary": "supports"} | j))
    monkeypatch.setattr(llm, "structured", lambda *a, **k: (out, llm.OFFLINE))
    return filing_rag.judge_llm(C(text="Services grew strongly", claim_type="qualitative"), [HIT])[0]


def test_judge_llm_accepts_verbatim_quote(monkeypatch):
    ev = _judge(monkeypatch)
    assert ev.stance == "supports" and ev.citations[0].quote == "Services grew strongly."


def test_judge_llm_rejects_non_verbatim_quote(monkeypatch):
    assert _judge(monkeypatch, quote="Services grew very strongly.").stance == "insufficient"
    assert _judge(monkeypatch, passage=5).stance == "insufficient"


def test_judge_llm_summary_is_audited(monkeypatch):
    assert "false" not in _judge(monkeypatch, summary="The claim is false.").summary
    ev = _judge(monkeypatch, summary="Services grew 40% to $31 billion.")  # uncited numbers
    assert ev.stance == "supports" and ev.summary == "Judgment on the retrieved passages: supports."


NUM_HIT = Hit(1, 9, "MD&A", "Total net sales were $94,930 million. Services revenue was $26.6 billion.",
              1.0, 0, 70)
Q1 = {9: ("10-Q", 2025, "Q1")}


def _judge_num(monkeypatch, claim=None, periods=Q1, **j):
    from ledgerline import llm
    out = filing_rag._Judgment(**({"stance": "contradicts", "passage": 0,
                                   "quote": "Total net sales were $94,930 million.",
                                   "summary": "contradicts"} | j))
    monkeypatch.setattr(llm, "structured", lambda *a, **k: (out, llm.OFFLINE))
    claim = claim or C(text="net sales of $94.9 billion", value=94.9e9, unit="USD",
                       metric="revenue")
    return filing_rag.judge_llm(claim, [NUM_HIT], None, periods)[0]


def test_judge_llm_numeric_contradiction_needs_its_period(monkeypatch):
    assert _judge_num(monkeypatch).stance == "contradicts"
    assert _judge_num(monkeypatch, periods={9: ("10-Q", 2024, "Q1")}).stance == "insufficient"
    assert _judge_num(monkeypatch, periods={}).stance == "insufficient"  # period unknown


@pytest.mark.parametrize("j", [
    {"quote": "Services revenue was $26.6 billion."},  # a different number
    {"passage": -1},
    {"quote": ""},
])
def test_judge_llm_negatives_are_insufficient(monkeypatch, j):
    assert _judge_num(monkeypatch, **j).stance == "insufficient"


def test_scoped_claim_is_not_contradicted_by_a_company_total(monkeypatch):
    scoped = C(text="revenue was $94.9 billion", value=94.9e9, unit="USD", metric=None)
    assert _judge_num(monkeypatch, scoped).stance == "insufficient"
    assert _judge_num(monkeypatch, scoped, passage_is_company_total=False).stance == "contradicts"


@pytest.mark.db
def test_online_run_sends_release_context_and_periods(db, monkeypatch):
    from datetime import UTC, datetime
    from pathlib import Path
    from threading import Lock

    from ledgerline import llm
    from ledgerline.graph.state import Ctx
    from ledgerline.models import Company, Filing

    text = (Path(__file__).parent / "fixtures" / "ex991_nvda_q2fy27.txt").read_text()
    monkeypatch.setattr("ledgerline.ingest.documents.load_text", lambda f: text)
    filing_rag._release_text.cache_clear()
    co = Company(cik="0001045810", ticker="NVDA", name="NVIDIA", fiscal_year_end="0131")
    db.add(co)
    db.flush()

    def filing(acc, form, path):
        f = Filing(company_id=co.id, accession_no=acc, form_type=form, filed_at=date(2026, 8, 26),
                   fiscal_year=2027, fiscal_period="Q2", doc_url="u", storage_path=path,
                   source_url="u", fetched_at=datetime.now(UTC), sha256="0" * 64, license_note="t")
        db.add(f)
        db.flush()
        return f
    release, q = filing("r1", "8-K", "objects/test-l6-release.htm"), filing("q1", "10-Q", "objects/q")
    phrase = "Data Center revenue of $89.0 billion"
    start = text.index(phrase)
    claim = Claim(company_id=co.id, fiscal_year=2027, fiscal_period="Q2", filing_id=release.id,
                  span_start=start, span_end=start + len(phrase), text=phrase,
                  claim_type="reported_result", value=89e9, unit="USD", extractor_version="t")
    hit = Hit(1, q.id, "MD&A", "Data Center revenue was $89.0 billion.", 1.0, 0, 30)
    monkeypatch.setattr(filing_rag, "_search", lambda *a: [hit])
    monkeypatch.setattr(llm, "available", lambda: True)
    seen = []

    def structured(tier, system, data, instruction, schema):
        seen.append(data)
        return schema(stance="supports", passage=0, quote="Data Center revenue was $89.0 billion.",
                      summary="supports"), llm.OFFLINE
    monkeypatch.setattr(llm, "structured", structured)
    ctx = Ctx(co.id, co.cik, co.ticker, None, None, Lock())
    ev, _ = filing_rag.run(db, claim, ctx, ctx.db_lock)
    assert ev.stance == "supports"
    assert phrase in seen[0]["claim_sentence"] and "passage 0 (10-Q fiscal 2027 Q2)" in seen[0]
    claim.span_start += 1  # the span no longer matches the text: no context rather than wrong
    filing_rag.run(db, claim, ctx, ctx.db_lock)
    assert not any(k.startswith(("claim_sentence", "claim_section")) for k in seen[1])
    filing_rag._release_text.cache_clear()

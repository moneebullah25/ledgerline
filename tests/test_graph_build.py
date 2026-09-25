import threading
from datetime import date
from threading import Lock

import pytest

from ledgerline import llm
from ledgerline.graph import adjudicator, build, critic
from ledgerline.graph.state import Ctx
from ledgerline.llm import OFFLINE, Usage
from ledgerline.models import Claim
from ledgerline.schemas import Citation, CriticOutput, EvidenceItem, VerdictDraft
from ledgerline.tracing import Tracer

CLAIM = Claim(id=7, company_id=1, fiscal_year=2025, fiscal_period="Q1", text="revenue of $94.9 billion",
              span_start=0, span_end=1, claim_type="reported_result", filing_id=1, metric="revenue",
              value=94.9e9, unit="USD", target_fiscal_year=2025, target_fiscal_period="Q1")
FACT = Citation(kind="xbrl_fact", ref_id=11, value=94_930e6, unit="USD")


def ev(agent, stance, citations=(), **payload):
    return EvidenceItem(agent=agent, stance=stance, summary=f"{agent} {stance}",
                        citations=list(citations), payload=payload)


@pytest.fixture
def fakes(monkeypatch):
    """Replace agents with fakes returning given evidence; no DB, no models."""
    monkeypatch.setattr(build, "context", lambda s, c: Ctx(1, "1", "AAPL", None, None, Lock()))

    def install(**by_agent):
        for name, item in by_agent.items():
            monkeypatch.setitem(build.AGENTS, name, lambda s, c, ctx, lock, item=item: (item, OFFLINE))
    return install


def test_supported_numeric_claim_carries_fact_ids(fakes):
    fakes(xbrl_table=ev("xbrl_table", "supports", [FACT], periods=[[2025, "Q1"]], tol_used=0.1),
          filing_rag=ev("filing_rag", "insufficient"))
    t = Tracer()
    r = build.verify_claim(None, CLAIM, t)
    assert r.verdict.label == "supported" and not r.refused
    assert [c.ref_id for c in r.verdict.citations] == [11]
    assert "94.93 billion" in r.verdict.rationale
    assert [e.agent for e in r.evidence] == ["xbrl_table", "filing_rag"]
    assert r.prompt_version.startswith("adjudicator-") and len(r.prompt_version) == 20
    names = {s.name for s in t.spans}
    assert {"verify_claim", "router", "agent.xbrl_table", "agent.filing_rag", "refusal_check",
            "critic", "adjudicator"} <= names
    root = next(s for s in t.spans if s.name == "verify_claim")
    assert all(s.parent_id == root.id for s in t.spans if s.name.startswith("agent."))


def test_refusal_short_circuits_without_critic_or_adjudicator(fakes, monkeypatch):
    calls = []
    monkeypatch.setattr(adjudicator, "adjudicate", lambda *a: calls.append("adj"))
    monkeypatch.setattr(critic, "critique", lambda *a: calls.append("critic"))
    fakes(xbrl_table=ev("xbrl_table", "insufficient"), filing_rag=ev("filing_rag", "insufficient"))
    r = build.verify_claim(None, CLAIM, Tracer())
    assert calls == []
    assert r.refused and r.verdict.label == "unverifiable" and r.critic is None


def test_market_reaction_alone_cannot_prevent_refusal(fakes, monkeypatch):
    monkeypatch.setattr(build, "context",
                        lambda s, c: Ctx(1, "1", "AAPL", date(2025, 1, 30), None, Lock()))
    price = Citation(kind="price", ref_id=1, value=200.0)
    fakes(xbrl_table=ev("xbrl_table", "insufficient"), filing_rag=ev("filing_rag", "insufficient"),
          market_reaction=ev("market_reaction", "insufficient", [price], abnormal_return=-0.05))
    r = build.verify_claim(None, CLAIM, Tracer())
    assert r.refused and len(r.evidence) == 3


def test_agents_run_concurrently(fakes, monkeypatch):
    barrier = threading.Barrier(2, timeout=5)  # deadlocks (BrokenBarrierError) if sequential

    def agent(name, stance="insufficient"):
        def run(s, c, ctx, lock):
            barrier.wait()
            return ev(name, stance), OFFLINE
        return run
    fakes()
    monkeypatch.setitem(build.AGENTS, "xbrl_table", agent("xbrl_table"))
    monkeypatch.setitem(build.AGENTS, "filing_rag", agent("filing_rag"))
    assert build.verify_claim(None, CLAIM, Tracer()).refused


def test_uncited_number_fails_closed(fakes, monkeypatch):
    fakes(xbrl_table=ev("xbrl_table", "supports", [FACT], periods=[[2025, "Q1"]]),
          filing_rag=ev("filing_rag", "insufficient"))
    monkeypatch.setattr(adjudicator, "_describe", lambda c, e: "Revenue was 97.2 billion.")
    with pytest.raises(adjudicator.IntegrityError, match="uncited"):
        build.verify_claim(None, CLAIM, Tracer())


def test_banned_vocabulary_fails_closed(fakes, monkeypatch):
    fakes(xbrl_table=ev("xbrl_table", "contradicts", [FACT], periods=[[2025, "Q1"]]),
          filing_rag=ev("filing_rag", "insufficient"))
    monkeypatch.setattr(adjudicator, "_describe", lambda c, e: "The claim is misleading.")
    with pytest.raises(adjudicator.IntegrityError, match="banned"):
        build.verify_claim(None, CLAIM, Tracer())


@pytest.mark.parametrize("word", ["misleading", "False", "fraudulent", "lie", "lies"])
def test_audit_flags_each_banned_word(word):
    assert adjudicator.audit(f"This is {word}.", [], "")


def test_generated_text_never_uses_banned_words(fakes):
    """Every offline template: critic arguments, refusal and adjudicator rationales."""
    texts = list(critic.ARGUMENTS.values())
    fakes(xbrl_table=ev("xbrl_table", "insufficient"), filing_rag=ev("filing_rag", "insufficient"))
    texts.append(build.verify_claim(None, CLAIM, Tracer()).verdict.rationale)
    for stance in ("supports", "contradicts"):
        for agent, cites, payload in [
            ("xbrl_table", [FACT], {"periods": [[2025, "Q1"]], "derived": True}),
            ("filing_rag", [Citation(kind="chunk", ref_id=3, quote="q")], {"margin": 0.1}),
        ]:
            evidence = [ev(agent, stance, cites, **payload)]
            out, _ = critic.critique(CLAIM, evidence)
            v, *_ = adjudicator.adjudicate(CLAIM, evidence, out)
            texts += [out.argument, v.rationale]
    assert not [t for t in texts if adjudicator.BANNED.search(t)]


def test_xbrl_dominates_numeric_and_dissent_lowers_confidence():
    chunk = Citation(kind="chunk", ref_id=3, quote="Net sales were $94.9 billion.")
    alone = [ev("xbrl_table", "supports", [FACT], periods=[[2025, "Q1"]])]
    both = alone + [ev("filing_rag", "contradicts", [chunk], margin=0.9)]
    v1, *_ = adjudicator.adjudicate(CLAIM, alone, None)
    v2, *_ = adjudicator.adjudicate(CLAIM, both, critic.critique(CLAIM, both)[0])
    assert v1.label == v2.label == "supported"
    assert v2.confidence < v1.confidence


GUIDANCE = Claim(id=8, company_id=1, fiscal_year=2025, fiscal_period="Q1", claim_type="guidance",
                 text="revenue between $89 billion and $93 billion", span_start=0, span_end=1,
                 filing_id=1, metric="revenue", value=89e9, value_high=93e9, unit="USD",
                 target_fiscal_year=2025, target_fiscal_period="Q2")
FORECAST = [Citation(kind="forecast", ref_id=f"revenue:2025Q2:ensemble:{k}", value=v, unit="USD")
            for k, v in (("point", 91e9), ("lo90", 86e9), ("hi90", 96e9))]
CHUNK = Citation(kind="chunk", ref_id=3, quote="Net sales were $94.9 billion.")


def test_filing_rag_support_prevents_refusal_when_xbrl_is_insufficient(fakes, monkeypatch):
    calls = []
    real = adjudicator.adjudicate
    monkeypatch.setattr(adjudicator, "adjudicate", lambda *a: calls.append(1) or real(*a))
    fakes(xbrl_table=ev("xbrl_table", "insufficient"),
          filing_rag=ev("filing_rag", "supports", [CHUNK], margin=0.9, section="MD&A"))
    r = build.verify_claim(None, CLAIM, Tracer())
    assert calls == [1] and not r.refused
    assert r.verdict.label == "supported" and r.verdict.confidence <= 0.6  # NLI on a number


def test_forecast_only_guidance_is_capped_at_unverifiable(fakes):
    fakes(forecast=ev("forecast", "supports", FORECAST, mase={"ensemble": 0.5}, edge=0.4),
          filing_rag=ev("filing_rag", "insufficient"))
    r = build.verify_claim(None, GUIDANCE, Tracer())
    assert not r.refused and r.verdict.label == "unverifiable"
    assert {c.kind for c in r.verdict.citations} == {"forecast"}  # still visible
    assert "86.00 billion" in r.verdict.rationale


def test_qualitative_claim_through_the_graph(fakes):
    claim = Claim(id=9, company_id=1, fiscal_year=2025, fiscal_period="Q1", filing_id=1,
                  claim_type="qualitative", text="Services grew strongly", span_start=0,
                  span_end=1)
    quote = Citation(kind="chunk", ref_id=4, quote="Services grew strongly.")
    fakes(filing_rag=ev("filing_rag", "supports", [quote], margin=0.5, section="MD&A"))
    t = Tracer()
    r = build.verify_claim(None, claim, t)
    assert r.verdict.label == "supported" and [e.agent for e in r.evidence] == ["filing_rag"]
    assert "agent.xbrl_table" not in {s.name for s in t.spans}


# ---- online adjudicator with a stubbed llm.structured

XBRL = ev("xbrl_table", "supports", [FACT], periods=[[2025, "Q1"]])
USAGE = Usage("claude-opus-5", 100, 10)


NEEDS_JUDGMENT = adjudicator.needs_judgment


@pytest.fixture
def online(monkeypatch):
    """Queue model outputs; returns the list of instructions the model received."""
    seen = []

    def install(*outputs):
        queue = list(outputs)
        monkeypatch.setattr(llm, "available", lambda: True)
        # Exercise the judge path; the gating rule has its own test below.
        monkeypatch.setattr(adjudicator, "needs_judgment", lambda claim, evidence: True)

        def structured(tier, system, data, instruction, schema):
            seen.append((instruction, data))
            return queue.pop(0), USAGE
        monkeypatch.setattr(llm, "structured", structured)
        return seen
    return install


def V(label="supported", rationale="Supported by filed revenue of 94.93 billion USD.", cites=(FACT,)):
    return VerdictDraft(label=label, confidence=0.9, rationale=rationale, citations=list(cites))


def test_online_bad_then_good(online):
    seen = online(V(rationale="Revenue was 97.2 billion.", cites=[FACT]), V())
    v, u, model = adjudicator.adjudicate(CLAIM, [XBRL], None)
    assert v.label == "supported" and u.input_tokens == 200 and model == "claude-opus-5"
    instruction, data = seen[1]
    assert "97.2" not in instruction and "uncited numbers" in instruction
    assert "97.2" in data["rejected_answer_problems"]


def test_online_both_bad_falls_back_to_offline_with_usage(online):
    bad = V(rationale="The claim is misleading.")
    online(bad, bad)
    v, u, model = adjudicator.adjudicate(CLAIM, [XBRL], None)
    assert model is None and u.input_tokens == 200  # rule verdict; both calls still priced
    assert v.label == "supported" and "misleading" not in v.rationale
    assert adjudicator.audit(v.rationale, v.citations, CLAIM.text) == []


def test_online_fabricated_citation_value_is_replaced_and_caught(online):
    fake = Citation(kind="xbrl_fact", ref_id=11, value=97.2e9, unit="USD")
    unknown = Citation(kind="xbrl_fact", ref_id=999, value=1.0)
    online(V(rationale="Filed revenue was 97.2 billion USD.", cites=[fake]),
           V(cites=[fake, unknown, fake]))
    v, *_ = adjudicator.adjudicate(CLAIM, [XBRL], None)
    assert v.citations == [FACT]  # the evidence's copy; unknown dropped; deduped


def test_online_label_must_match_deciding_stance(online):
    seen = online(V(label="contradicted"), V(label="contradicted"))
    derived = ev("xbrl_table", "supports", [FACT], periods=[[2025, "Q1"]], derived=True)
    online_critic = CriticOutput(counter_stance="contradicts", argument="Revenue was 97.2 billion.",
                                 weakest_evidence="x")
    v, u, model = adjudicator.adjudicate(CLAIM, [derived], online_critic)
    assert len(seen) == 2 and "label mismatch" in seen[1][0]
    assert v.label == "supported" and "94.93 billion" in v.rationale  # offline rule
    assert "97.2" not in v.rationale and critic.ARGUMENTS["xbrl_table:derived_fact"] in v.rationale
    assert model is None and u.model == "claude-opus-5" and u.input_tokens == 200  # spend kept


def test_online_label_mismatch_with_other_problems_falls_back(online):
    bad = V(label="contradicted", rationale="Revenue was 97.2 billion.")
    online(bad, bad)
    v, u, model = adjudicator.adjudicate(CLAIM, [XBRL], None)
    assert v.label == "supported" and model is None and u.input_tokens == 200


def test_online_fallback_integrity_error_carries_usage(online, monkeypatch):
    online(V(label="contradicted"), V(label="contradicted"))
    monkeypatch.setattr(adjudicator, "audit", lambda text, *a: ["uncited numbers: [1]"])
    with pytest.raises(adjudicator.IntegrityError) as exc:
        adjudicator.adjudicate(CLAIM, [XBRL], None)
    assert exc.value.usage.input_tokens == 200


def test_online_forecast_only_must_be_unverifiable(online):
    online(V(label="supported", cites=FORECAST[:1], rationale="Consistent with history."),
           V(label="unverifiable", cites=FORECAST[:1], rationale="Consistent with history only."))
    forecast = ev("forecast", "supports", FORECAST, mase={"ensemble": 0.5})
    v, *_ = adjudicator.adjudicate(GUIDANCE, [forecast], None)
    assert v.label == "unverifiable"


def test_online_critic_retries_on_uncited_number_then_falls_back(online):
    online(CriticOutput(counter_stance="contradicts", argument="Revenue was really 97.2 billion."),
           CriticOutput(counter_stance="contradicts", argument="The fact is derived."))
    out, _ = critic.critique(CLAIM, [XBRL])
    assert out.argument == "The fact is derived."
    bad = CriticOutput(counter_stance="contradicts", argument="This is a lie.")
    seen = online(bad, bad)
    out, u = critic.critique(CLAIM, [XBRL])
    assert out == critic.critique_offline(CLAIM, [XBRL]) and u.input_tokens > 0
    assert "rejected_answer_problems" in seen[-1][1]  # the retry says why


def test_judge_model_only_when_judgment_is_needed(online, monkeypatch):
    seen = online(V(), V())
    monkeypatch.setattr(adjudicator, "needs_judgment", NEEDS_JUDGMENT)
    v, _, model = adjudicator.adjudicate(CLAIM, [XBRL], None)  # unopposed filed fact
    assert model is None and seen == [] and v.label == "supported"
    rag = ev("filing_rag", "contradicts", [Citation(kind="chunk", ref_id=5, quote="q")])
    assert adjudicator.needs_judgment(CLAIM, [XBRL, rag])  # agents disagree
    assert not adjudicator.needs_judgment(CLAIM, [XBRL])


def test_offline_rationale_drops_a_critic_argument_it_cannot_cite():
    # A live critic may cite another agent's number (a price move); the rule's rationale
    # cites only the deciding agent, so it names the point instead of quoting it.
    loose = CriticOutput(counter_stance="contradicts", argument="Shares moved 8.4% after the release.",
                         weakest_evidence="market_reaction:context")
    v, _, _ = adjudicator.adjudicate(CLAIM, [XBRL], loose)
    assert "8.4" not in v.rationale and "market_reaction:context" in v.rationale

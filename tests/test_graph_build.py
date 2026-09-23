import threading
from datetime import date
from threading import Lock

import pytest

from ledgerline.graph import adjudicator, build, critic
from ledgerline.graph.state import Ctx
from ledgerline.llm import OFFLINE
from ledgerline.models import Claim
from ledgerline.schemas import Citation, EvidenceItem
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


def test_generated_text_never_uses_banned_words():
    """Every offline template: critic arguments, refusal and adjudicator rationales."""
    texts = list(critic.ARGUMENTS.values())
    for stance in ("supports", "contradicts"):
        for agent, cites, payload in [
            ("xbrl_table", [FACT], {"periods": [[2025, "Q1"]], "derived": True}),
            ("filing_rag", [Citation(kind="chunk", ref_id=3, quote="q")], {"margin": 0.1}),
        ]:
            evidence = [ev(agent, stance, cites, **payload)]
            out, _ = critic.critique(CLAIM, evidence)
            v, _ = adjudicator.adjudicate(CLAIM, evidence, out)
            texts += [out.argument, v.rationale]
    assert not [t for t in texts if adjudicator.BANNED.search(t)]


def test_xbrl_dominates_numeric_and_dissent_lowers_confidence():
    chunk = Citation(kind="chunk", ref_id=3, quote="Net sales were $94.9 billion.")
    alone = [ev("xbrl_table", "supports", [FACT], periods=[[2025, "Q1"]])]
    both = alone + [ev("filing_rag", "contradicts", [chunk], margin=0.9)]
    v1, _ = adjudicator.adjudicate(CLAIM, alone, None)
    v2, _ = adjudicator.adjudicate(CLAIM, both, critic.critique(CLAIM, both)[0])
    assert v1.label == v2.label == "supported"
    assert v2.confidence < v1.confidence

"""verify_claim: router -> concurrent evidence agents -> refusal check -> critic -> adjudicator,
as a LangGraph StateGraph.

The agents are fanned out with Send, so LangGraph runs them as one superstep on its thread
pool. They share the caller's Session only under Ctx.db_lock. The Tracer is not thread-safe
(its span stack is shared), so each agent records into a child Tracer seeded with the
parent's open span and the spans are merged back when the agent finishes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from threading import Lock

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline import llm
from ledgerline.graph import adjudicator, critic, load_prompt
from ledgerline.graph.agents import filing_rag, forecast, market_reaction, xbrl_table
from ledgerline.graph.router import period_key, route
from ledgerline.graph.state import Ctx, GraphState
from ledgerline.models import CallAudio, Claim, Company, Filing, Segment
from ledgerline.schemas import AgentName, CriticOutput, EvidenceItem, VerdictDraft
from ledgerline.tracing import Tracer

AgentFn = Callable[[Session, Claim, Ctx, Lock], tuple[EvidenceItem, llm.Usage]]
AGENTS: dict[AgentName, AgentFn] = {
    "xbrl_table": xbrl_table.run,
    "filing_rag": filing_rag.run,
    "forecast": forecast.run,
    "market_reaction": market_reaction.run,
}
_ORDER = list(AGENTS)


@dataclass
class VerifyResult:
    verdict: VerdictDraft
    evidence: list[EvidenceItem]
    critic: CriticOutput | None
    refused: bool
    prompt_version: str
    model: str | None
    cost_usd: float
    usages: list[tuple[str, llm.Usage]]


def _record(attrs: dict, u: llm.Usage) -> None:
    attrs |= {"model": u.model, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
              "cache_read_tokens": u.cache_read_tokens, "cost_usd": u.cost_usd}


def _router(state: GraphState, config) -> dict:
    ctx = state["ctx"]
    with config["configurable"]["tracer"].span("router") as a:
        agents = route(state["claim"], event_date=ctx.event_date,
                       reported_through=ctx.reported_through)
        a["agents"] = agents
    return {"agents": agents}


def _fan_out(state: GraphState) -> list[Send]:
    return [Send(name, state) for name in state["agents"]]


def _agent_node(name: AgentName):
    def node(state: GraphState, config) -> dict:
        parent: Tracer = config["configurable"]["tracer"]
        t = Tracer(parent.trace_id)
        t._stack = list(parent._stack)  # parent the span under verify_claim
        try:
            with t.span(f"agent.{name}") as a:
                ev, usage = AGENTS[name](config["configurable"]["session"], state["claim"],
                                         state["ctx"], state["ctx"].db_lock)
                _record(a, usage)
                a["stance"] = ev.stance
        finally:
            parent.spans.extend(t.spans)
        return {"evidence": [ev], "usages": [(name, usage)]}
    return node


def _refusal_check(state: GraphState, config) -> dict:
    """FR-44: no truth-bearing evidence either way -> unverifiable, no critic, no adjudicator."""
    with config["configurable"]["tracer"].span("refusal_check") as a:
        truth = [e for e in state["evidence"] if e.agent in adjudicator.TRUTH_AGENTS]
        refused = all(e.stance == "insufficient" for e in truth)
        a["refused"] = refused
        if not refused:
            return {"refused": False}
        checked = ", ".join(e.agent for e in truth) or "none"
        rationale = ("Unverifiable. No truth-bearing agent found evidence that supports or "
                     f"contradicts the claim (checked: {checked}).")
        if problems := adjudicator.audit(rationale, [], state["claim"].text):
            raise adjudicator.IntegrityError(f"refusal rationale failed audit: {problems}")
        # Confidence 1.0: the refusal is a mechanical fact about the evidence, not a judgment.
        verdict = VerdictDraft(label="unverifiable", confidence=1.0, rationale=rationale,
                               citations=[])
        return {"refused": True, "verdict": verdict, "critic": None, "model": None}


def _critic(state: GraphState, config) -> dict:
    with config["configurable"]["tracer"].span("critic") as a:
        out, usage = critic.critique(state["claim"], state["evidence"])
        _record(a, usage)
        a["counter_stance"], a["weakest_evidence"] = out.counter_stance, out.weakest_evidence
    return {"critic": out, "usages": [("critic", usage)]}


def _adjudicator(state: GraphState, config) -> dict:
    with config["configurable"]["tracer"].span("adjudicator") as a:
        verdict, usage = adjudicator.adjudicate(state["claim"], state["evidence"], state["critic"])
        _record(a, usage)
        a["label"], a["confidence"] = verdict.label, verdict.confidence
    return {"verdict": verdict, "model": usage.model, "usages": [("adjudicator", usage)]}


@cache
def graph():
    g = StateGraph(GraphState)
    g.add_node("router", _router)
    for name in AGENTS:
        g.add_node(name, _agent_node(name))
        g.add_edge(name, "refusal_check")
    g.add_node("refusal_check", _refusal_check)
    g.add_node("critic", _critic)
    g.add_node("adjudicator", _adjudicator)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", _fan_out, list(AGENTS))
    g.add_conditional_edges("refusal_check", lambda st: END if st["refused"] else "critic",
                            ["critic", END])
    g.add_edge("critic", "adjudicator")
    g.add_edge("adjudicator", END)
    return g.compile()


def context(s: Session, claim: Claim) -> Ctx:
    company = s.get(Company, claim.company_id)
    event_date = None
    if claim.filing_id is not None:
        event_date = s.get(Filing, claim.filing_id).filed_at
    elif claim.segment_id is not None:
        event_date = s.scalar(select(CallAudio.event_date).join(
            Segment, Segment.call_audio_id == CallAudio.id).where(Segment.id == claim.segment_id))
    periods = s.execute(select(Filing.fiscal_year, Filing.fiscal_period).where(
        Filing.company_id == claim.company_id, Filing.form_type.in_(filing_rag.PERIODIC),
        Filing.fiscal_year.is_not(None), Filing.fiscal_period.is_not(None))).all()
    reported = max(periods, key=lambda p: period_key(*p), default=None)
    return Ctx(company.id, company.cik, company.ticker, event_date,
               tuple(reported) if reported else None, Lock())


def verify_claim(s: Session, claim: Claim, tracer: Tracer) -> VerifyResult:
    # Load every column now, on this thread: an expired attribute would otherwise lazy-load
    # through the Session from inside an agent thread, outside db_lock.
    for col in Claim.__table__.columns:
        getattr(claim, col.key)
    with tracer.span("verify_claim", claim_id=claim.id) as a:
        out = graph().invoke(
            {"claim": claim, "ctx": context(s, claim), "evidence": [], "usages": []},
            config={"configurable": {"session": s, "tracer": tracer}},
        )
        usages = out["usages"]
        cost = sum(u.cost_usd for _, u in usages)
        a |= {"label": out["verdict"].label, "refused": out["refused"], "cost_usd": cost}
    return VerifyResult(
        verdict=out["verdict"],
        evidence=sorted(out["evidence"], key=lambda e: _ORDER.index(e.agent)),
        critic=out.get("critic"),
        refused=out["refused"],
        prompt_version=load_prompt("adjudicator")[1],
        model=out.get("model"),
        cost_usd=cost,
        usages=usages,
    )

"""FR-42 critic: the strongest counter-argument to the leading stance. It never labels.

Offline it points at the weakest part of the evidence, in this order: a truth-bearing agent
that disagrees; an XBRL match at the edge of the rounding tolerance; a derived fact; a
period assumed because the claim named none; a low NLI margin; a forecast with barely any
skill over seasonal naive or a guided value at the interval edge. The argument text is
built without numbers so it can be quoted in a rationale without new citations.
"""

import json

from ledgerline import llm
from ledgerline.graph import load_prompt
from ledgerline.graph.adjudicator import (
    TRUTH_AGENTS,
    add_usage,
    audit,
    decide,
)
from ledgerline.models import Claim
from ledgerline.schemas import CriticOutput, EvidenceItem

_FLIP = {"supports": "contradicts", "contradicts": "supports"}


ARGUMENTS = {
    "xbrl_table:rounding_edge": (
        "The gap between the claim and the filed fact is close to the tolerance boundary; "
        "a slightly different rounding convention would flip the comparison."),
    "xbrl_table:derived_fact": (
        "The filed figure is derived from component facts rather than reported directly, "
        "so a definitional difference is possible."),
    "xbrl_table:assumed_period": (
        "The claim does not name its period; the comparison assumes it refers to the "
        "quarter of the release."),
    "filing_rag:low_nli_margin": (
        "The entailment model barely preferred this reading of the passage over the "
        "alternatives."),
    "forecast:weak_skill": "The forecast only narrowly beats seasonal naive on held-out quarters.",
    "forecast:interval_edge": "The guided value sits near the edge of the forecast interval.",
}


def _weakest(claim: Claim, evidence: list[EvidenceItem], stance: str) -> tuple[str, str] | None:
    """(code, argument) for the weakest point in the leading evidence, or None."""
    for e in evidence:
        if e.agent in TRUTH_AGENTS and e.stance == _FLIP[stance]:
            return f"{e.agent}:disagrees", f"The {e.agent} evidence reaches the opposite stance."
    by = {e.agent: e for e in evidence if e.stance == stance}
    code = None
    if x := by.get("xbrl_table"):
        if 0.8 <= x.payload.get("tol_used", 0) <= 1.25:
            code = "xbrl_table:rounding_edge"
        elif x.payload.get("derived"):
            code = "xbrl_table:derived_fact"
        elif claim.target_fiscal_period is None:
            code = "xbrl_table:assumed_period"
    if code is None and (r := by.get("filing_rag")) and r.payload.get("margin", 1.0) < 0.3:
        code = "filing_rag:low_nli_margin"
    if code is None and (f := by.get("forecast")):
        if f.payload["mase"]["ensemble"] > 0.8:
            code = "forecast:weak_skill"
        elif f.payload.get("edge", 1.0) < 0.1:
            code = "forecast:interval_edge"
    return (code, ARGUMENTS[code]) if code else None


def critique(claim: Claim, evidence: list[EvidenceItem]) -> tuple[CriticOutput, llm.Usage]:
    stance, _ = decide(claim, evidence)
    if llm.available():
        return _critique_llm(claim, evidence, stance)
    return critique_offline(claim, evidence), llm.OFFLINE


def critique_offline(claim: Claim, evidence: list[EvidenceItem]) -> CriticOutput:
    stance, _ = decide(claim, evidence)
    weak = _weakest(claim, evidence, stance)
    if weak is None:
        return CriticOutput(counter_stance=_FLIP[stance], weakest_evidence=None,
                            argument="No weak point found in the leading evidence.")
    code, argument = weak
    return CriticOutput(counter_stance=_FLIP[stance], argument=argument, weakest_evidence=code)


def _critique_llm(claim, evidence, stance) -> tuple[CriticOutput, llm.Usage]:
    system, _ = load_prompt("critic")
    data = {"claim": claim.text, "leading_stance": stance,
            "evidence": json.dumps([e.model_dump() for e in evidence])}
    cites = [c for e in evidence for c in e.citations]
    total = llm.Usage()
    instruction = "Argue the other side."
    for _ in range(2):
        out, usage = llm.structured("standard", system, data, instruction, CriticOutput)
        total = add_usage(total, usage)
        # The argument can be quoted in the rationale: same number and vocabulary rules.
        if not (problems := audit(out.argument, cites, claim.text)):
            return out, total
        # Details quote model output, so they go in a data block, not the instruction (A6).
        data = data | {"rejected_answer_problems": json.dumps(problems)}
        instruction = ("Argue the other side. Your previous argument was rejected; the "
                       "rejected_answer_problems document lists why.")
    # The critic is advisory: an argument that keeps failing the audit is replaced by the
    # deterministic offline critic instead of costing the claim its verdict. Spend is kept.
    return critique_offline(claim, evidence), total

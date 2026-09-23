"""FR-43 adjudicator: evidence + critique -> VerdictDraft.

Offline decision rule (deterministic, used when llm.available() is False):
1. The deciding agent is the first non-`insufficient` truth-bearing agent in
   TRUTH_AGENTS order. The XBRL typed query dominates numeric claims; FilingRAG (NLI) decides claims
   without a filed fact; the forecast agent is considered only for guidance.
2. Label = supported / contradicted from that agent's stance.
3. Confidence starts from the agent (XBRL 0.9, 0.8 if the fact is derived; forecast 0.55;
   NLI 0.5 + 0.4 * margin, capped at 0.6 on a numeric claim) and drops 0.15 if another
   truth-bearing agent took the opposite stance and 0.1 if the critic found a weak point.

Either way FR-45 is enforced: every number in the rationale must trace to a citation (or
the claim), and the §16 vocabulary rule bans "misleading", "false", "fraudulent", "lie".
Offline a violation is a bug and raises; online the model gets one retry, then it raises.
"""

import json
import re

from ledgerline import llm
from ledgerline.graph import load_prompt
from ledgerline.models import Claim
from ledgerline.schemas import (
    Citation,
    CriticOutput,
    EvidenceItem,
    Stance,
    VerdictDraft,
    uncited_numbers,
)

TRUTH_AGENTS = ("xbrl_table", "forecast", "filing_rag")
LABEL = {"supports": "supported", "contradicts": "contradicted"}
BANNED = re.compile(r"\b(misleading\w*|false\w*|fraud\w*|lie|lies|lied|lying|liar\w*)\b", re.IGNORECASE)


class IntegrityError(RuntimeError):
    """Generated text failed citation integrity (FR-45) or the vocabulary rule (§16)."""


def audit(text: str, citations: list[Citation], claim_text: str) -> list[str]:
    problems = []
    if nums := uncited_numbers(text, citations, claim_text):
        problems.append(f"uncited numbers: {nums}")
    if words := BANNED.findall(text):
        problems.append(f"banned words: {words}")
    return problems


def decide(claim: Claim, evidence: list[EvidenceItem]) -> tuple[Stance, EvidenceItem | None]:
    """The leading stance and the evidence item that decides it (step 1 of the rule)."""
    by = {e.agent: e for e in evidence}
    for name in TRUTH_AGENTS:  # dominance order
        e = by.get(name)
        if e and e.stance != "insufficient" and (name != "forecast" or claim.claim_type == "guidance"):
            return e.stance, e
    return "insufficient", None


def fmt(v: float, unit: str | None) -> str:
    """Print a cited value so that schemas.uncited_numbers recognises it."""
    if unit == "USD" and abs(v) >= 1e9:
        return f"{v / 1e9:.2f} billion USD"
    if unit == "USD" and abs(v) >= 1e6:
        return f"{v / 1e6:.2f} million USD"
    return f"{v:.2f}{'%' if unit == 'percent' else ' ' + unit if unit else ''}"


def _describe(claim: Claim, e: EvidenceItem) -> str:
    verb = "is consistent with" if e.stance == "supports" else "is inconsistent with"
    if e.agent == "xbrl_table":
        seen, facts = set(), []
        for c in e.citations:
            if c.value not in seen:
                seen.add(c.value)
                facts.append(fmt(c.value, c.unit))
        periods = " and ".join(f"{fy} {fp}" for fy, fp in e.payload["periods"])
        kind = f"{claim.comparison} change" if len(e.payload["periods"]) > 1 else "value"
        derived = " (derived from component facts)" if e.payload.get("derived") else ""
        return (f"Filed XBRL {claim.metric} for {periods}: {', '.join(facts)}{derived}. "
                f"The claimed {kind} {verb} these facts within the rounding-aware tolerance.")
    if e.agent == "forecast":
        vals = {c.ref_id.rsplit(":", 1)[1]: fmt(c.value, c.unit) for c in e.citations}
        return (f"The guided value {verb} the ninety-percent interval implied by quarterly "
                f"history ({vals['lo90']} to {vals['hi90']}). This measures consistency with "
                "history, not the eventual outcome.")
    return (f"A cited passage in {e.payload.get('section', 'the filings')} "
            f"{'entails' if e.stance == 'supports' else 'contradicts'} the claim.")


def adjudicate(
    claim: Claim, evidence: list[EvidenceItem], critic: CriticOutput | None,
) -> tuple[VerdictDraft, llm.Usage]:
    if llm.available():
        return _adjudicate_llm(claim, evidence, critic)
    stance, e = decide(claim, evidence)
    if e is None:  # build.py refuses before this; reaching here is a wiring bug
        raise ValueError("adjudicator called without decisive evidence")
    if e.agent == "xbrl_table":
        conf = 0.8 if e.payload.get("derived") else 0.9
    elif e.agent == "forecast":
        conf = 0.55
    else:
        conf = 0.5 + 0.4 * e.payload.get("margin", 0.0)
        if claim.value is not None:
            conf = min(conf, 0.6)
    dissent = [x for x in evidence if x.agent in TRUTH_AGENTS and x is not e
               and x.stance not in ("insufficient", stance)]
    parts = [_describe(claim, e)]
    if dissent:
        conf -= 0.15
        parts.append(f"The {', '.join(x.agent for x in dissent)} evidence points the other way; "
                     f"the {e.agent} evidence takes precedence under the dominance rule.")
    if critic and critic.weakest_evidence:
        conf -= 0.1
        parts.append(f"Critic: {critic.argument}")
    label = LABEL[stance]
    rationale = f"{label.capitalize()}. " + " ".join(parts)
    cites = list(e.citations)
    if problems := audit(rationale, cites, claim.text):
        raise IntegrityError(f"offline rationale failed audit: {problems}")
    return VerdictDraft(label=label, confidence=round(min(max(conf, 0.05), 0.99), 3),
                        rationale=rationale, citations=cites), llm.OFFLINE


def _adjudicate_llm(claim, evidence, critic) -> tuple[VerdictDraft, llm.Usage]:
    system, _ = load_prompt("adjudicator")
    data = {"claim": claim.text,
            "evidence": json.dumps([e.model_dump() for e in evidence]),
            "critic": critic.model_dump_json() if critic else "none"}
    known = {(c.kind, str(c.ref_id)) for e in evidence for c in e.citations}
    instruction = "Adjudicate the claim."
    total = llm.Usage()
    for _ in range(2):  # FR-45: fail closed after one regeneration
        v, u = llm.structured("judge", system, data, instruction, VerdictDraft)
        total = add_usage(total, u)
        problems = audit(v.rationale, v.citations, claim.text)
        if bad := [c.ref_id for c in v.citations if (c.kind, str(c.ref_id)) not in known]:
            problems.append(f"citations not in the evidence: {bad}")
        if not problems:
            return v, total
        instruction = f"Adjudicate the claim. Your previous answer was rejected: {problems}"
    raise IntegrityError(f"adjudicator output failed audit twice: {problems}")


def add_usage(a: llm.Usage, b: llm.Usage) -> llm.Usage:
    return llm.Usage(b.model, a.input_tokens + b.input_tokens, a.output_tokens + b.output_tokens,
                     a.cache_read_tokens + b.cache_read_tokens,
                     a.cache_write_tokens + b.cache_write_tokens)

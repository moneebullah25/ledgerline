"""FR-43 adjudicator: evidence + critique -> VerdictDraft.

Offline decision rule (deterministic, used when llm.available() is False):
1. The deciding agent is the first non-`insufficient` truth-bearing agent in
   TRUTH_AGENTS order. The XBRL typed query dominates numeric claims; FilingRAG (NLI) decides claims
   without a filed fact; the forecast agent is considered only for guidance.
2. Label = supported / contradicted from that agent's stance, except that the forecast agent
   alone never verifies anything: guidance decided only by it is `unverifiable` (the
   forecast evidence is still cited and described).
3. Confidence starts from the agent (XBRL 0.9, 0.8 if the fact is derived; forecast 0.55;
   NLI 0.5 + 0.4 * margin, capped at 0.6 on a numeric claim) and drops 0.15 if another
   truth-bearing agent took the opposite stance and 0.1 if the critic found a weak point.

Either way FR-45 is enforced: every number in the rationale must trace to a citation (or
the claim), and the §16 vocabulary rule bans "misleading", "false", "fraudulent", "lie".
Offline a violation is a bug and raises; online the model gets one retry, then it raises.
Online, every citation the model returns is replaced by the evidence citation with the same
(kind, ref_id), so it cannot supply its own values or quotes; unknown ones are dropped. The
model's label must equal the rule's label (expected_label); if it still disagrees after the
retry, the offline verdict is returned instead.
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
    Label,
    Stance,
    VerdictDraft,
    uncited_numbers,
)

TRUTH_AGENTS = ("xbrl_table", "forecast", "filing_rag")
LABEL = {"supports": "supported", "contradicts": "contradicted"}
BANNED = re.compile(r"\b(misleading\w*|false\w*|fraud\w*|lie|lies|lied|lying|liar\w*)\b", re.IGNORECASE)


class IntegrityError(RuntimeError):
    """Generated text failed citation integrity (FR-45) or the vocabulary rule (§16).
    `usage` carries the model spend of the failed attempts (NFR-3), None offline."""

    def __init__(self, message: str, usage: llm.Usage | None = None):
        super().__init__(message)
        self.usage = usage


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


def expected_label(claim: Claim, evidence: list[EvidenceItem]) -> Label:
    """The label the decision rule gives; the online adjudicator must agree with it."""
    stance, e = decide(claim, evidence)
    return "unverifiable" if e is None or e.agent == "forecast" else LABEL[stance]


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
        def values(cs):
            return ", ".join(dict.fromkeys(fmt(c.value, c.unit) for c in cs))
        computed = [c for c in e.citations if c.kind == "computed"]
        tol = next((c for c in computed if str(c.ref_id).endswith(":tolerance")), None)
        change = [c for c in computed if str(c.ref_id).endswith("_change")]
        ratio = [c for c in computed if c is not tol and c not in change]
        periods = " and ".join(f"{fy} {fp}" for fy, fp in e.payload["periods"])
        kind = f"{claim.comparison} change" if change else "value"
        derived = " (derived from other filed facts)" if e.payload.get("derived") else ""
        text = (f"Filed XBRL facts for {claim.metric}, {periods}: "
                f"{values(c for c in e.citations if c.kind == 'xbrl_fact')}{derived}.")
        if ratio:
            text += f" Computed {claim.metric}: {values(ratio)}."
        if change:
            text += f" Computed {kind}: {values(change)}."
        of = f" of {fmt(tol.value, tol.unit)}" if tol else ""
        how = "is within" if e.stance == "supports" else "differs from these facts by more than"
        return text + f" The claimed {kind} {how} the rounding-aware tolerance{of}."
    if e.agent == "forecast":
        vals = {c.ref_id.rsplit(":", 1)[1]: fmt(c.value, c.unit) for c in e.citations}
        return (f"The guided value {verb} the ninety-percent interval implied by quarterly "
                f"history ({vals['lo90']} to {vals['hi90']}). This measures consistency with "
                "history, not the eventual outcome, so on its own it does not verify the claim.")
    return (f"A cited passage in {e.payload.get('section', 'the filings')} "
            f"{'entails' if e.stance == 'supports' else 'contradicts'} the claim.")


def adjudicate(
    claim: Claim, evidence: list[EvidenceItem], critic: CriticOutput | None,
) -> tuple[VerdictDraft, llm.Usage]:
    if llm.available():
        return _adjudicate_llm(claim, evidence, critic)
    return _adjudicate_offline(claim, evidence, critic), llm.OFFLINE


def _adjudicate_offline(
    claim: Claim, evidence: list[EvidenceItem], critic: CriticOutput | None,
) -> VerdictDraft:
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
    label = expected_label(claim, evidence)
    rationale = f"{label.capitalize()}. " + " ".join(parts)
    cites = list(e.citations)
    if problems := audit(rationale, cites, claim.text):
        raise IntegrityError(f"offline rationale failed audit: {problems}")
    return VerdictDraft(label=label, confidence=round(min(max(conf, 0.05), 0.99), 3),
                        rationale=rationale, citations=cites)


def _adjudicate_llm(claim, evidence, critic) -> tuple[VerdictDraft, llm.Usage]:
    system, _ = load_prompt("adjudicator")
    data = {"claim": claim.text,
            "evidence": json.dumps([e.model_dump() for e in evidence]),
            "critic": critic.model_dump_json() if critic else "none"}
    known = {(c.kind, str(c.ref_id)): c for e in evidence for c in e.citations}
    want = expected_label(claim, evidence)
    instruction = "Adjudicate the claim."
    total = llm.Usage()
    for _ in range(2):  # FR-45: fail closed after one regeneration
        v, u = llm.structured("judge", system, data, instruction, VerdictDraft)
        total = add_usage(total, u)
        # A1: the evidence's own copy of each citation, never the model's value or quote.
        keys = dict.fromkeys((c.kind, str(c.ref_id)) for c in v.citations)
        cites = [known[k] for k in keys if k in known]
        problems = audit(v.rationale, cites, claim.text)
        if v.label != "unverifiable" and not cites:
            problems.append("no citations: none of the returned citations is in the evidence")
        if v.label != want:
            problems.append(f"label mismatch: {v.label} but the deciding evidence gives {want}")
        if not problems:
            return v.model_copy(update={"citations": cites}), total
        # A6: problem details quote model output, so they go in a data block, not the instruction.
        kinds = sorted({p.split(":")[0] for p in problems})
        data = data | {"rejected_answer_problems": json.dumps(problems)}
        instruction = (f"Adjudicate the claim. Your previous answer was rejected for "
                       f"{len(problems)} problem(s) of kind {', '.join(kinds)}; the "
                       "rejected_answer_problems document lists them.")
    if all(p.startswith("label mismatch") for p in problems):
        return _adjudicate_offline(claim, evidence, critic), total
    raise IntegrityError(f"adjudicator output failed audit twice: {problems}", total)


def add_usage(a: llm.Usage, b: llm.Usage) -> llm.Usage:
    return llm.Usage(b.model, a.input_tokens + b.input_tokens, a.output_tokens + b.output_tokens,
                     a.cache_read_tokens + b.cache_read_tokens,
                     a.cache_write_tokens + b.cache_write_tokens)

"""FilingRAG agent: does the company's own 10-K/10-Q text entail the claim?

Only periodic filings are searched: the 8-K release the claim came from would trivially
entail itself. Offline, an NLI cross-encoder scores (passage window, claim) pairs over the
top reranked chunks; each window is two consecutive sentences, an exact substring of the
chunk, which becomes the citation quote. Online, Sonnet judges the same chunks.

Numbers: for a claim with a value, only windows that print that number (at any scale,
within the claim's rounding) may support or contradict it. Otherwise this agent would be a
backdoor for reading numbers out of prose, which FR-22 reserves for the XBRL agent.
"""

import re
from functools import cache
from itertools import pairwise
from threading import Lock

import numpy as np
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline import llm
from ledgerline.config import settings
from ledgerline.graph import load_prompt
from ledgerline.graph.adjudicator import BANNED
from ledgerline.graph.agents.xbrl_table import mentions_value
from ledgerline.graph.state import Ctx
from ledgerline.models import Claim, Filing
from ledgerline.schemas import Citation, EvidenceItem, Stance
from ledgerline.store.retrieval import Hit, search

K = 5  # reranked chunks judged
ENTAIL_MIN = 0.6
# NLI models over-call contradiction between topically close but different statements
# (another segment, another period), so contradiction needs a higher bar than entailment.
CONTRA_MIN = 0.8
PERIODIC = ("10-K", "10-Q", "10-K/A", "10-Q/A")
_SENT_END = re.compile(r"(?<=[.!?])\s+")
# cross-encoder/nli-deberta-v3-small config.id2label: 0 contradiction, 1 entailment, 2 neutral
_CONTRA, _ENTAIL = 0, 1


@cache
def _model():
    from sentence_transformers import CrossEncoder

    m = CrossEncoder(settings().nli_model)
    labels = m.model.config.id2label
    assert labels[_CONTRA] == "contradiction" and labels[_ENTAIL] == "entailment", labels
    return m


def nli(pairs: list[tuple[str, str]]) -> np.ndarray:
    """(premise, hypothesis) -> softmax probs, columns in the model's id2label order."""
    logits = np.asarray(_model().predict(pairs))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def windows(text: str) -> list[str]:
    """Two-sentence windows as exact substrings of `text`."""
    cuts = [0] + [m.end() for m in _SENT_END.finditer(text)] + [len(text)]
    spans = [(a, b) for a, b in pairwise(cuts) if text[a:b].strip()]
    if len(spans) < 2:
        return [text.strip()] if text.strip() else []
    return [text[spans[i][0]:spans[i + 1][1]].strip() for i in range(len(spans) - 1)]


def run(s: Session, claim: Claim, ctx: Ctx, lock: Lock) -> tuple[EvidenceItem, llm.Usage]:
    with lock:
        fids = list(s.scalars(select(Filing.id).where(
            Filing.company_id == ctx.company_id, Filing.form_type.in_(PERIODIC))))
        hits = search(s, claim.text, company_id=ctx.company_id, k=K, filing_ids=fids) if fids else []
    if not hits:
        return _insufficient("no 10-K/10-Q passages retrieved for this company"), llm.OFFLINE
    if llm.available():
        return judge_llm(claim, hits)
    return assess(claim, hits, nli), llm.OFFLINE


def _insufficient(why: str, **payload) -> EvidenceItem:
    return EvidenceItem(agent="filing_rag", stance="insufficient", summary=why, payload=payload)


def assess(claim: Claim, hits: list[Hit], score=nli) -> EvidenceItem:
    cands = [(h, w) for h in hits for w in windows(h.text)]
    if claim.value is not None:
        cands = [(h, w) for h, w in cands if mentions_value(w, claim)]
        if not cands:
            return _insufficient("no retrieved passage prints the claimed figure")
    probs = score([(w, claim.text) for _, w in cands])
    ie, ic = int(probs[:, _ENTAIL].argmax()), int(probs[:, _CONTRA].argmax())
    pe, pc = float(probs[ie, _ENTAIL]), float(probs[ic, _CONTRA])
    if pe >= ENTAIL_MIN and pe >= pc:
        stance, i, label = "supports", ie, _ENTAIL
    elif pc >= CONTRA_MIN:
        stance, i, label = "contradicts", ic, _CONTRA
    else:
        return _insufficient("no passage entails or contradicts the claim",
                             p_entail=pe, p_contra=pc, windows_scored=len(cands))
    h, w = cands[i]
    margin = float(probs[i, label] - np.delete(probs[i], label).max())
    return EvidenceItem(
        agent="filing_rag", stance=stance,
        summary=f"NLI: a passage in {h.section_path} {'entails' if stance == 'supports' else 'contradicts'} the claim.",
        citations=[Citation(kind="chunk", ref_id=h.chunk_id, quote=w)],
        payload={"p_entail": pe, "p_contra": pc, "margin": margin,
                 "section": h.section_path, "windows_scored": len(cands)},
    )


class _Judgment(BaseModel):
    stance: Stance
    passage: int  # index into the passages given
    quote: str  # verbatim span from that passage
    summary: str


def judge_llm(claim: Claim, hits: list[Hit]) -> tuple[EvidenceItem, llm.Usage]:
    system, version = load_prompt("filing_rag")
    data = {"claim": claim.text} | {f"passage {i}": h.text for i, h in enumerate(hits)}
    j, usage = llm.structured("standard", system, data, "Judge the claim against the passages.",
                              _Judgment)
    ok = 0 <= j.passage < len(hits) and j.quote and j.quote in hits[j.passage].text
    if BANNED.search(j.summary):  # §16 vocabulary: keep the stance, drop the model's wording
        j.summary = f"The cited passage {'supports' if j.stance == 'supports' else 'contradicts'} the claim."
    if j.stance == "insufficient" or not ok or (
            claim.value is not None and not mentions_value(j.quote, claim)):
        return _insufficient(j.summary, prompt_version=version), usage
    return EvidenceItem(
        agent="filing_rag", stance=j.stance, summary=j.summary,
        citations=[Citation(kind="chunk", ref_id=hits[j.passage].chunk_id, quote=j.quote)],
        payload={"prompt_version": version, "section": hits[j.passage].section_path},
    ), usage

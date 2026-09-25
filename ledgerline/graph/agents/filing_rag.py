"""FilingRAG agent: does the company's own 10-K/10-Q text entail the claim?

Only periodic filings are searched: the 8-K release the claim came from would trivially
entail itself. Offline, an NLI cross-encoder scores (passage window, claim) pairs over the
top reranked chunks; each window is two consecutive sentences, an exact substring of the
chunk, which becomes the citation quote. Online, Sonnet judges the same chunks. Offline it only supports or abstains (see below).

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
from ledgerline.graph.adjudicator import audit
from ledgerline.graph.agents.xbrl_table import mentions_value
from ledgerline.graph.state import Ctx
from ledgerline.models import Claim, Company, Filing
from ledgerline.schemas import Citation, EvidenceItem, Stance
from ledgerline.store import embed
from ledgerline.store.retrieval import Hit, search

K = 5  # reranked chunks judged
ENTAIL_MIN = 0.6
# Offline, this agent never returns `contradicts`. On the AAPL/MSFT integration run the
# small NLI model called contradiction at p > 0.95 between a CEO quote and an unrelated
# segment sentence, and between "$143.8 billion" and a table printing $143,756 million:
# every contradiction inspected was spurious. The contradiction score is kept in the
# payload for the critic; only the LLM path may contradict.
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
    with embed.INFERENCE_LOCK:
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


def _search(s: Session, query: str, company_id: int, lock: Lock) -> list[Hit]:
    """Hybrid search over the company's 10-K/10-Q chunks; `lock` covers only DB reads, so
    the other agents are not serialised behind embedding and rerank inference."""
    with lock:
        fids = list(s.scalars(select(Filing.id).where(
            Filing.company_id == company_id, Filing.form_type.in_(PERIODIC))))
    return search(s, query, company_id=company_id, k=K, filing_ids=fids, db_lock=lock) if fids else []


def run(s: Session, claim: Claim, ctx: Ctx, lock: Lock) -> tuple[EvidenceItem, llm.Usage]:
    hits = _search(s, claim.text, ctx.company_id, lock)
    if not hits:
        return _insufficient("no 10-K/10-Q passages retrieved for this company"), llm.OFFLINE
    if llm.available():
        return judge_llm(claim, hits, claim_context(s, claim, lock))
    return assess(claim, hits, nli), llm.OFFLINE


def claim_context(s: Session, claim: Claim, lock: Lock) -> dict[str, str]:
    """The release text around the claim (section heading, neighbouring sentences): the claim
    span alone can drop its scope, e.g. "revenue was $89.0 billion" under a Data Center
    heading. Call claims (segment_id) carry no filing text, so they get none."""
    if claim.filing_id is None:
        return {}
    from ledgerline.extract.refine import context  # extract pulls in the NER stack
    from ledgerline.ingest.documents import load_text

    with lock:
        filing = s.get(Filing, claim.filing_id)
        fye = s.get(Company, claim.company_id).fiscal_year_end
        text = load_text(filing)
    ctx = context(text, claim, claim.fiscal_year, claim.fiscal_period, fye)
    return {f"claim_{k}": v for k, v in ctx.items() if k != "claim" and v}


def _insufficient(why: str, **payload) -> EvidenceItem:
    return EvidenceItem(agent="filing_rag", stance="insufficient", summary=why, payload=payload)


def assess(claim: Claim, hits: list[Hit], score=nli) -> EvidenceItem:
    cands = [(h, w) for h in hits for w in windows(h.text)]
    if claim.value is not None:
        cands = [(h, w) for h, w in cands if mentions_value(w, claim)]
        if not cands:
            return _insufficient("no retrieved passage prints the claimed figure")
    probs = score([(w, claim.text) for _, w in cands])
    ie = int(probs[:, _ENTAIL].argmax())
    pe, pc = float(probs[ie, _ENTAIL]), float(probs[:, _CONTRA].max())
    if pe < ENTAIL_MIN or pe < probs[ie, _CONTRA]:
        return _insufficient("no passage entails the claim",
                             p_entail=pe, p_contra=pc, windows_scored=len(cands))
    h, w = cands[ie]
    margin = float(pe - np.delete(probs[ie], _ENTAIL).max())
    return EvidenceItem(
        agent="filing_rag", stance="supports",
        summary=f"NLI: a passage in {h.section_path} entails the claim.",
        citations=[Citation(kind="chunk", ref_id=h.chunk_id, quote=w)],
        payload={"p_entail": pe, "p_contra": pc, "margin": margin,
                 "section": h.section_path, "windows_scored": len(cands)},
    )


class _Judgment(BaseModel):
    stance: Stance
    passage: int  # index into the passages given
    quote: str  # verbatim span from that passage
    summary: str


def judge_llm(claim: Claim, hits: list[Hit],
              scope: dict[str, str] | None = None) -> tuple[EvidenceItem, llm.Usage]:
    system, version = load_prompt("filing_rag")
    data = ({"claim": claim.text} | (scope or {})
            | {f"passage {i}": h.text for i, h in enumerate(hits)})
    j, usage = llm.structured("standard", system, data, "Judge the claim against the passages.",
                              _Judgment)
    ok = 0 <= j.passage < len(hits) and j.quote and j.quote in hits[j.passage].text
    cites = [Citation(kind="chunk", ref_id=hits[j.passage].chunk_id, quote=j.quote)] if ok else []
    if audit(j.summary, cites, claim.text):  # FR-45 / §16: keep the stance, drop the wording
        j.summary = f"Judgment on the retrieved passages: {j.stance}."
    if j.stance == "insufficient" or not ok or (
            claim.value is not None and not mentions_value(j.quote, claim)):
        return _insufficient(j.summary, prompt_version=version), usage
    return EvidenceItem(
        agent="filing_rag", stance=j.stance, summary=j.summary, citations=cites,
        payload={"prompt_version": version, "section": hits[j.passage].section_path},
    ), usage

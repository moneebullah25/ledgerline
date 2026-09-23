"""FR-21 hybrid search over `chunk`: full-text + dense, fused with RRF, optional rerank.

The sparse leg is Postgres full-text ranked by ts_rank (term frequency). It is BM25-style
in role (lexical, rewards exact terms and numbers) but not BM25: there is no IDF term and
no length normalization. The contract named ts_rank_cd; on the L1 eval (150 constructed
queries, same OR query) ts_rank_cd gave sparse Recall@10 0.467 vs 0.620 for ts_rank, so
ts_rank is used. Numbers survive the english parser: "94,930" becomes the phrase
'94' <-> '930' in both the tsvector and the query, and "94.9" stays one token, so a query
quoting a printed figure matches it exactly.
"""

import re
from dataclasses import dataclass
from typing import NamedTuple

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from ledgerline.store import embed
from ledgerline.store.index import passage

LEG_DEPTH = 50  # candidates per leg
RERANK_DEPTH = 30
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    filing_id: int
    section_path: str
    text: str
    score: float
    char_start: int
    char_end: int


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


_WORD = re.compile(r"\w+(?:[.,]\w+)*")


def _or_query(query: str) -> str:
    """websearch_to_tsquery ANDs every word, which almost never matches a whole natural
    question. OR-ing the words lets ts_rank score by how often they occur.
    Only word tokens are kept, so user text cannot inject websearch operators (-, quotes)."""
    return " or ".join(w for w in _WORD.findall(query) if w.lower() not in ("or", "and"))


def _filters(filing_ids: list[int] | None) -> str:
    # [] means "no filings in scope" (returns nothing), not "no filter".
    return "f.company_id = :cid" + ("" if filing_ids is None else " AND c.filing_id = ANY(:fids)")


def sparse_ids(s: Session, query: str, company_id: int,
               filing_ids: list[int] | None = None, n: int = LEG_DEPTH) -> list[int]:
    sql = f"""
        SELECT c.id FROM chunk c JOIN filing f ON f.id = c.filing_id,
             websearch_to_tsquery('english', :q) q
        WHERE {_filters(filing_ids)} AND c.tsv @@ q
        ORDER BY ts_rank(c.tsv, q) DESC, c.id LIMIT :n"""
    params = dict(q=_or_query(query), cid=company_id, n=n, fids=filing_ids)
    return list(s.scalars(text(sql), params))


def dense_ids(s: Session, qvec, company_id: int,
              filing_ids: list[int] | None = None, n: int = LEG_DEPTH) -> list[int]:
    # hnsw stops after ef_search candidates; with a company filter most of those belong to
    # other companies. pgvector >= 0.8 iterative scans keep walking the graph until n
    # filtered rows are found, in exact distance order.
    s.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    s.execute(text("SET LOCAL hnsw.ef_search = 100"))
    sql = f"""
        SELECT c.id FROM chunk c JOIN filing f ON f.id = c.filing_id
        WHERE {_filters(filing_ids)} AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :n"""
    params = dict(q=str(list(map(float, qvec))), cid=company_id, n=n, fids=filing_ids)
    return list(s.scalars(text(sql), params))


class _Row(NamedTuple):
    filing_id: int
    section_path: str
    text: str
    char_start: int
    char_end: int
    passage: str  # what the reranker scores, see index.passage


def _load(s: Session, ids: list[int]) -> dict[int, _Row]:
    if not ids:
        return {}
    rows = s.execute(
        text("""SELECT c.id, c.filing_id, c.section_path, c.text, c.char_start, c.char_end,
                       co.name, f.form_type, f.fiscal_year, f.fiscal_period, f.period_of_report
                FROM chunk c JOIN filing f ON f.id = c.filing_id
                JOIN company co ON co.id = f.company_id
                WHERE c.id IN :ids""").bindparams(bindparam("ids", expanding=True)),
        {"ids": ids},
    )
    return {r.id: _Row(r.filing_id, r.section_path, r.text, r.char_start, r.char_end,
                       passage(r.name, r.form_type, r.fiscal_year, r.fiscal_period,
                               r.period_of_report, r.section_path, r.text))
            for r in rows}


def search(s: Session, query: str, *, company_id: int, k: int = 10,
           filing_ids: list[int] | None = None, rerank: bool = True) -> list[Hit]:
    fused = rrf([
        sparse_ids(s, query, company_id, filing_ids),
        dense_ids(s, embed.embed_query(query), company_id, filing_ids),
    ])
    scored = fused[:RERANK_DEPTH] if rerank else fused[:k]
    rows = _load(s, [cid for cid, _ in scored])
    if rerank and scored:
        # The reranker sees the same company/period header as the embedder. Its logits
        # replace the RRF score and are only comparable within one query.
        logits = embed.rerank_scores(query, [rows[cid].passage for cid, _ in scored])
        scored = sorted(zip((cid for cid, _ in scored), logits), key=lambda x: -x[1])
    return [
        Hit(cid, r.filing_id, r.section_path, r.text, float(score), r.char_start, r.char_end)
        for cid, score in scored[:k]
        for r in [rows[cid]]
    ]

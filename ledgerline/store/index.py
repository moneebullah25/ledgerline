"""Chunk + embed filings into the `chunk` table (FR-20)."""

from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ledgerline.models import Chunk, Company, Filing
from ledgerline.store import embed
from ledgerline.store.chunking import chunk_document


def passage(company: str, form_type: str, fiscal_year: int | None, fiscal_period: str | None,
            period_end: date | None, section_path: str, text: str) -> str:
    """What the embedder and the reranker see: the chunk behind a header naming company,
    form, period and section. A statement-table chunk rarely says whose it is or which
    quarter it covers, and every quarter's filing reads alike. chunk.text itself stays an
    exact slice of the document."""
    return (f"{company} {form_type} fiscal {fiscal_year} {fiscal_period}, period ended "
            f"{period_end} | {section_path}\n{text}")


def index_filing(s: Session, filing_id: int) -> int:
    """(Re)index one filing; returns chunks written.

    Upserts on (filing_id, chunk_index), so a re-run never duplicates and chunk ids stay
    stable for citations; chunks beyond the new count (after a chunker change) are removed.
    """
    from ledgerline.ingest.documents import load_text  # owned by S1; imported lazily

    filing = s.get(Filing, filing_id)
    if filing is None:
        raise ValueError(f"no filing {filing_id}")
    drafts = chunk_document(load_text(filing), filing.form_type)
    if drafts:
        name = s.get(Company, filing.company_id).name
        vectors = embed.embed_passages([
            passage(name, filing.form_type, filing.fiscal_year, filing.fiscal_period,
                    filing.period_of_report, d.section_path, d.text)
            for d in drafts
        ])
        rows = [
            dict(filing_id=filing_id, chunk_index=d.chunk_index, section_path=d.section_path,
                 char_start=d.char_start, char_end=d.char_end, text=d.text,
                 token_count=d.token_count, embedding=v)
            for d, v in zip(drafts, vectors)
        ]
        stmt = insert(Chunk).values(rows)
        s.execute(stmt.on_conflict_do_update(
            constraint="uq_chunk",
            set_={c: stmt.excluded[c] for c in rows[0] if c not in ("filing_id", "chunk_index")},
        ))
    s.execute(delete(Chunk).where(Chunk.filing_id == filing_id,
                                  Chunk.chunk_index >= len(drafts)))
    return len(drafts)


def unindexed_filing_ids(s: Session) -> list[int]:
    has_chunks = select(Chunk.filing_id).where(Chunk.filing_id == Filing.id).exists()
    return list(s.scalars(select(Filing.id).where(~has_chunks).order_by(Filing.id)))

"""python -m ledgerline.store index --all | index --filing ID | explain | verify-offsets"""

import argparse
import re
import sys
import time

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ledgerline.db import session
from ledgerline.models import Chunk, Claim, Filing, Segment
from ledgerline.store import embed
from ledgerline.store.index import index_filing, unindexed_filing_ids

# Unfiltered top-50 nearest neighbours: the query shape hnsw serves directly.
_DENSE = "SELECT id FROM chunk ORDER BY embedding <=> CAST(:q AS vector) LIMIT 50"
# The dense leg as search() runs it: company filter through the filing join.
_FILTERED = """SELECT c.id FROM chunk c JOIN filing f ON f.id = c.filing_id
    WHERE f.company_id = (SELECT min(id) FROM company) AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT 50"""


def explain(query: str = "total net sales for the quarter") -> None:
    """EXPLAIN ANALYZE the dense leg: hnsw vs a forced sequential scan, then the filtered
    production query. Each is run twice and the second (warm cache) plan is printed."""
    q = str(list(map(float, embed.embed_query(query))))
    runs = [
        ("hnsw, unfiltered", _DENSE, ["SET LOCAL hnsw.ef_search = 100"]),
        ("seqscan, unfiltered", _DENSE, ["SET LOCAL enable_indexscan = off"]),
        ("search() dense leg, company filter", _FILTERED,
         ["SET LOCAL hnsw.ef_search = 100", "SET LOCAL hnsw.iterative_scan = strict_order"]),
    ]
    for label, sql, setup in runs:
        with session() as s:  # SET LOCAL ends with the transaction
            for stmt in setup:
                s.execute(text(stmt))
            for _ in range(2):
                plan = [r[0] for r in s.execute(text("EXPLAIN ANALYZE " + sql), {"q": q})]
        print(f"--- {label}\n" + "\n".join(re.sub(r"'\[[^']*\]'", "'[...]'", ln) for ln in plan))


def verify_offsets(s: Session) -> int:
    """Re-slice every chunk and claim against its source as loaded today (load_text for
    filings, segment.text for call claims, whose spans are segment-relative). Prints each
    mismatch; returns how many there were."""
    from ledgerline.ingest.documents import load_text

    bad = checked = 0

    def check(kind, rid, where, source, a, b, t):
        nonlocal bad, checked
        checked += 1
        if source is None or source[a:b] != t:
            bad += 1
            print(f"{kind} {rid} ({where}) [{a}:{b}] does not match")

    segments = dict(s.execute(select(Segment.id, Segment.text)).all())
    for seg_id, cid, a, b, t in s.execute(select(
            Claim.segment_id, Claim.id, Claim.span_start, Claim.span_end, Claim.text)
            .where(Claim.segment_id.is_not(None))):
        check("claim", cid, f"segment {seg_id}", segments[seg_id], a, b, t)
    for filing in s.scalars(select(Filing).order_by(Filing.id)):
        rows = [("chunk", *r) for r in s.execute(select(
            Chunk.id, Chunk.char_start, Chunk.char_end, Chunk.text)
            .where(Chunk.filing_id == filing.id))]
        rows += [("claim", *r) for r in s.execute(select(
            Claim.id, Claim.span_start, Claim.span_end, Claim.text)
            .where(Claim.filing_id == filing.id))]
        if rows:
            try:
                doc = load_text(filing)
            except OSError as e:  # object missing from the store: nothing can be re-sliced
                print(f"filing {filing.id}: cannot load text ({e})")
                doc = None
            for kind, rid, a, b, t in rows:
                check(kind, rid, f"filing {filing.id}", doc, a, b, t)
    print(f"verified {checked} offsets: {bad} mismatches")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ledgerline.store")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index")
    g = ix.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="every filing without chunks yet")
    g.add_argument("--filing", type=int)
    sub.add_parser("explain")
    sub.add_parser("verify-offsets", help="exit 1 if any chunk/claim no longer re-slices")
    a = p.parse_args()

    if a.cmd == "verify-offsets":
        with session() as s:
            sys.exit(1 if verify_offsets(s) else 0)

    if a.cmd == "explain":
        explain()
        return
    with session() as s:
        ids = unindexed_filing_ids(s) if a.all else [a.filing]
    t0, total = time.perf_counter(), 0
    for fid in ids:
        with session() as s:  # one transaction per filing: progress survives a crash
            n = index_filing(s, fid)
        total += n
        print(f"filing {fid}: {n} chunks")
    print(f"indexed {len(ids)} filings, {total} chunks in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

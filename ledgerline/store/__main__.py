"""python -m ledgerline.store index --all | index --filing ID | explain"""

import argparse
import re
import time

from sqlalchemy import text

from ledgerline.db import session
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


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ledgerline.store")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index")
    g = ix.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="every filing without chunks yet")
    g.add_argument("--filing", type=int)
    sub.add_parser("explain")
    a = p.parse_args()

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

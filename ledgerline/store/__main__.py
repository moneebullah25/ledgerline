"""python -m ledgerline.store index --all | index --filing ID | explain"""

import argparse
import time

from sqlalchemy import text

from ledgerline.db import session
from ledgerline.store import embed
from ledgerline.store.index import index_filing, unindexed_filing_ids

# Unfiltered top-50 nearest neighbours: the query shape hnsw serves directly.
_DENSE = "SELECT id FROM chunk ORDER BY embedding <=> CAST(:q AS vector) LIMIT 50"


def explain(query: str = "total net sales for the quarter") -> None:
    """EXPLAIN ANALYZE the dense leg with the hnsw index vs a forced sequential scan."""
    q = str(list(map(float, embed.embed_query(query))))
    with session() as s:
        for label, setup in (("hnsw", "SET LOCAL enable_seqscan = on"),
                             ("seqscan", "SET LOCAL enable_indexscan = off")):
            s.execute(text(setup))
            for _ in range(2):  # first run warms the buffer cache; report the second
                plan = [r[0] for r in s.execute(text("EXPLAIN ANALYZE " + _DENSE), {"q": q})]
            print(f"--- {label}\n" + "\n".join(plan))
            s.execute(text("RESET enable_indexscan"))


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

"""python -m ledgerline.ingest backfill (--ticker AAPL | --all) [--quarters 6]

Calls the plain functions in-process; no Prefect server is needed.
"""

import argparse
import logging
import time

from ledgerline.concepts import UNIVERSE
from ledgerline.ingest.flows import run_facts, run_filings, run_prices


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ledgerline.ingest")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="filings + XBRL facts + prices")
    who = b.add_mutually_exclusive_group(required=True)
    who.add_argument("--ticker", action="append", choices=list(UNIVERSE))
    who.add_argument("--all", action="store_true")
    b.add_argument("--quarters", type=int, default=6)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = list(UNIVERSE) if args.all else args.ticker
    t0 = time.monotonic()
    for name, result in (("filings", run_filings(tickers, args.quarters)),
                         ("facts", run_facts(tickers)),
                         ("prices", run_prices(tickers))):
        print(f"{name}: {result}  (inserted {sum(v or 0 for v in result.values())})", flush=True)
    print(f"elapsed {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()

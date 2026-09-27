"""Backfill: `python -m ledgerline.extract [--ticker NVDA] [--no-ner]`

Extracts claims from every 8-K earnings release and every transcribed earnings call not
yet extracted at the current extractor_version(), then links guidance to later actuals per company.
One transaction per document, so a failure loses only that document.
"""

import argparse
import time

from sqlalchemy import exists, select

from ledgerline.db import session
from ledgerline.extract.pipeline import (
    extract_for_call,
    extract_for_filing,
    extractor_version,
    link_across_quarters,
)
from ledgerline.models import CallAudio, Claim, Company, Filing, Segment


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ledgerline.extract")
    p.add_argument("--ticker")
    p.add_argument("--no-ner", action="store_true", help="regex/lexicon only, no GLiNER")
    a = p.parse_args()
    t0 = time.perf_counter()
    with session() as s:
        companies = s.scalars(
            select(Company.id).where(Company.ticker == a.ticker if a.ticker else True)
        ).all()
        filings = s.scalars(
            select(Filing.id)
            .where(
                Filing.form_type == "8-K",
                Filing.company_id.in_(companies),
                ~exists().where(
                    Claim.filing_id == Filing.id, Claim.extractor_version == extractor_version()
                ),
            )
            .order_by(Filing.id)
        ).all()
        calls = s.scalars(
            select(CallAudio.id).where(
                CallAudio.kind == "earnings_call",
                CallAudio.company_id.in_(companies),
                exists().where(Segment.call_audio_id == CallAudio.id),
            )
        ).all()
    for kind, ids, fn in (
        ("filing", filings, extract_for_filing),
        ("call", calls, extract_for_call),
    ):
        for i in ids:
            try:
                with session() as s:
                    print(f"{kind} {i}: {fn(s, i, use_ner=not a.no_ner)} claims", flush=True)
            except ValueError as e:  # no fiscal period on the source, or the LLM pass failed twice
                print(f"{kind} {i}: skipped ({e})", flush=True)
    with session() as s:
        linked = sum(link_across_quarters(s, c) for c in companies)
    print(f"linked {linked} actuals to guidance; {time.perf_counter() - t0:.1f}s total")


if __name__ == "__main__":
    main()

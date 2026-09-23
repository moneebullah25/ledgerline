"""L4 temporal backtest (PRD §11): do forecast-agent flags on guidance predict a later miss?

    python -m ledgerline.evals.l4 [--no-write]

Population: every extracted guidance claim with a canonical metric and a value (current
extractor version; not the golden set, which holds only 25 guidance claims).
At claim time: the forecast agent runs exactly as in verify_claim; it only sees quarterly
XBRL history up to the release quarter. `flagged` = stance `contradicts` (the guided value
is outside the 90% interval implied by history).
Later outcome: the XBRL typed query for the guided period, compared with the guided range
by the XBRL agent's own tolerance (`missed` = the filed actual landed outside it). Claims
whose target period has no filed fact yet have no outcome and are excluded.

backtest_precision = P(missed | flagged). It is only meaningful next to the base rate
P(missed | not flagged), reported alongside. Pairs = distinct (company, release quarter)
with at least one guidance claim whose outcome is known.

Known limits: the forecast agent covers quarterly targets only (fiscal-year guidance is
`insufficient`, so never flagged); history uses the earliest as-filed XBRL value per period
but a restated component can still leak; linked claims are not used, XBRL is the outcome.
"""

import argparse
import json
import subprocess
from threading import Lock

from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline.concepts import CONCEPTS
from ledgerline.graph.agents import forecast, xbrl_table
from ledgerline.graph.build import context
from ledgerline.models import Claim, Company, EvalRun

TARGET_PAIRS = 40
TARGET_PRECISION = 0.55


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(rows: list[dict]) -> dict:
    known = [r for r in rows if r["missed"] is not None]
    flagged = [r for r in known if r["flagged"]]
    unflagged = [r for r in known if not r["flagged"]]
    pairs = {(r["ticker"], r["release"]) for r in known}
    return {
        "backtest_precision": _rate(sum(r["missed"] for r in flagged), len(flagged)),
        "miss_rate_unflagged": _rate(sum(r["missed"] for r in unflagged), len(unflagged)),
        "miss_rate_all": _rate(sum(r["missed"] for r in known), len(known)),
        "flag_recall": _rate(sum(r["flagged"] for r in known if r["missed"]),
                             sum(r["missed"] for r in known)),
        "n_guidance": len(rows), "n_outcome_known": len(known), "n_flagged": len(flagged),
        "n_forecast_decided": sum(r["forecast_stance"] != "insufficient" for r in known),
        "pairs": len(pairs), "pairs_target_met": len(pairs) >= TARGET_PAIRS,
    }


def backtest(s: Session) -> list[dict]:
    claims = s.execute(
        select(Claim, Company.ticker).join(Company, Company.id == Claim.company_id)
        .where(Claim.claim_type == "guidance", Claim.value.is_not(None),
               Claim.metric.in_(list(CONCEPTS)))
        .order_by(Claim.id)
    ).all()
    rows = []
    for c, ticker in claims:
        ctx, lock = context(s, c), Lock()
        fc, _ = forecast.run(s, c, ctx, lock)
        out, _ = xbrl_table.run(s, c, ctx, lock)
        rows.append({
            "claim_id": c.id, "ticker": ticker, "release": f"{c.fiscal_year}{c.fiscal_period}",
            "target": f"{c.target_fiscal_year}{c.target_fiscal_period}", "metric": c.metric,
            "text": c.text, "forecast_stance": fc.stance, "forecast_why": fc.summary,
            "flagged": fc.stance == "contradicts",
            "missed": None if out.stance == "insufficient" else out.stance == "contradicts",
            "actual": out.payload.get("actual"), "extractor_version": c.extractor_version,
        })
        print(json.dumps({k: rows[-1][k] for k in ("claim_id", "ticker", "target", "metric",
                                                   "forecast_stance", "missed")}), flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    from ledgerline.db import session

    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.l4")
    p.add_argument("--no-write", action="store_true", help="skip the eval_run row")
    a = p.parse_args(argv)
    with session() as s:
        rows = backtest(s)
        metrics = summarize(rows)
        if not a.no_write:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                 check=False).stdout.strip() or None
            s.add(EvalRun(git_sha=sha, layer="L4", metrics=metrics, details={
                "targets": {"pairs": TARGET_PAIRS, "precision": TARGET_PRECISION},
                "rows": rows}))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

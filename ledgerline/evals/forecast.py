"""L1 forecast eval (S6): MASE vs seasonal naive, rolling origin over held-out quarters.

    python -m ledgerline.evals.forecast

For every universe company and concept, the forecast agent's backtest (one quarter ahead,
last HOLDOUT quarters held out) gives each model's MASE = held-out MAE / seasonal-naive
held-out MAE. forecast_mase is the arithmetic mean of the ensemble's per-series MASE;
the median and geometric mean are reported too because a ratio metric is skewed by series
where seasonal naive happens to be near-perfect. PRD target: < 1.0.
"""

import subprocess

import numpy as np

from ledgerline.concepts import UNIVERSE
from ledgerline.db import session
from ledgerline.graph.agents.forecast import HOLDOUT, MIN_TRAIN, backtest, contiguous_tail
from ledgerline.models import EvalRun
from ledgerline.store.xbrl_query import get_series

EVAL_CONCEPTS = ("revenue", "operating_income", "net_income")
H = 1


def main() -> None:
    with session() as s:
        rows, skipped = [], []
        for ticker, cik in UNIVERSE.items():
            for concept in EVAL_CONCEPTS:
                y = np.array([f.value for f in contiguous_tail(get_series(s, cik, concept))])
                if len(y) < HOLDOUT + MIN_TRAIN:
                    skipped.append({"ticker": ticker, "concept": concept, "n": len(y)})
                    continue
                rows.append({"ticker": ticker, "concept": concept, "n": len(y),
                             **backtest(y, H)})
                print(rows[-1], flush=True)
        if not rows:
            raise SystemExit(f"no series long enough ({len(skipped)} skipped)")
        models = [k for k in rows[0] if k not in ("ticker", "concept", "n")]
        ens = np.array([r["ensemble"] for r in rows])
        metrics = {
            "forecast_mase": float(ens.mean()),
            "forecast_mase_median": float(np.median(ens)),
            "forecast_mase_geomean": float(np.exp(np.log(np.maximum(ens, 1e-9)).mean())),
            "share_beating_naive": float((ens < 1.0).mean()),
            "series": len(rows),
        } | {f"mase_{m}": float(np.mean([r[m] for r in rows])) for m in models}
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout
        s.add(EvalRun(git_sha=sha.strip() or None, layer="L1", name="forecast", metrics=metrics,
                      details={"horizon": H, "holdout": HOLDOUT, "rows": rows,
                               "skipped": skipped}))
    print(metrics)


if __name__ == "__main__":
    main()

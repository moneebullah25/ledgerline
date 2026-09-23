"""Forecast agent (S6): is guidance consistent with the company's own quarterly history?

Semantics: `supports` = the guided value (midpoint of a range) lies inside the ensemble's
90% interval; `contradicts` = it lies outside. That is "consistent / inconsistent with
history", never "the guidance is untrue": the future has not happened yet.

Models: seasonal naive, AutoETS, AutoARIMA (statsforecast, season 4) and Chronos-Bolt
(CPU). Ensemble = mean of the ETS, ARIMA and Chronos point forecasts and interval bounds
(quantile averaging). Chronos-Bolt was trained on quantiles 0.1..0.9 only, so its 5%/95%
bounds are extrapolated from the 10%/90% ones assuming normal tails (x z95/z90).

Skill check (the S6 rule): rolling-origin backtest over the last HOLDOUT quarters at the
guidance horizon h. "MASE" here is held-out MAE of a model divided by held-out MAE of the
seasonal-naive forecast on the same points (out-of-sample scaling). If the ensemble does
not beat seasonal naive (MASE >= 1.0) the stance is `insufficient`: a forecast that
doesn't beat seasonal naive shouldn't be in the verdict.
"""

from functools import cache
from threading import Lock

import numpy as np
from sqlalchemy.orm import Session

from ledgerline.concepts import CONCEPTS
from ledgerline.graph.router import period_key
from ledgerline.graph.state import Ctx
from ledgerline.llm import OFFLINE, Usage
from ledgerline.models import Claim
from ledgerline.schemas import Citation, EvidenceItem
from ledgerline.store.xbrl_query import Fact, get_series

SEASON = 4
HOLDOUT = 4
MIN_TRAIN = 8
MAX_H = 4
ENSEMBLE = ("ets", "arima", "chronos")
CHRONOS_MODEL = "amazon/chronos-bolt-small"
_Z95_OVER_Z90 = 1.6449 / 1.2816


@cache
def _chronos():
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu")


def forecast_all(y: np.ndarray, h: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """name -> (mean, lo90, hi90), each of length h."""
    import torch
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

    out = {}
    for name, m in (("snaive", SeasonalNaive(SEASON)), ("ets", AutoETS(season_length=SEASON)),
                    ("arima", AutoARIMA(season_length=SEASON))):
        f = m.forecast(y=y, h=h, level=[90])
        out[name] = tuple(np.asarray(f[k], dtype=float) for k in ("mean", "lo-90", "hi-90"))
    q, _ = _chronos().predict_quantiles(torch.tensor(y, dtype=torch.float32), prediction_length=h,
                                        quantile_levels=[0.1, 0.5, 0.9])
    q10, q50, q90 = (q[0, :, i].numpy().astype(float) for i in range(3))
    out["chronos"] = (q50, q50 - (q50 - q10) * _Z95_OVER_Z90, q50 + (q90 - q50) * _Z95_OVER_Z90)
    out["ensemble"] = tuple(np.mean([out[n][i] for n in ENSEMBLE], axis=0) for i in range(3))
    return out


def mase(pred: np.ndarray, naive: np.ndarray, actual: np.ndarray) -> float:
    """Held-out MAE of `pred` over held-out MAE of the seasonal-naive forecast."""
    return float(np.mean(np.abs(pred - actual)) / max(np.mean(np.abs(naive - actual)), 1e-12))


def backtest(y: np.ndarray, h: int, holdout: int = HOLDOUT) -> dict[str, float]:
    """Rolling origin: each of the last `holdout` points is forecast h steps ahead from an
    origin that ends h points before it. Returns name -> MASE vs seasonal naive."""
    preds: dict[str, list[float]] = {}
    for t in range(len(y) - holdout, len(y)):
        for name, (mean, _, _) in forecast_all(y[: t - h + 1], h).items():
            preds.setdefault(name, []).append(mean[-1])
    actual = y[-holdout:]
    naive = np.array(preds["snaive"])
    return {n: mase(np.array(p), naive, actual) for n, p in preds.items() if n != "snaive"}


def _qi(fiscal_year: int, fiscal_period: str) -> int:
    """Quarter index: consecutive quarters differ by 1."""
    return fiscal_year * 4 + period_key(fiscal_year, fiscal_period)[1]


def contiguous_tail(facts: list[Fact]) -> list[Fact]:
    """The longest run of consecutive quarters ending at the last fact: seasonal models
    assume no gaps."""
    out = facts[-1:]
    for f in reversed(facts[:-1]):
        if _qi(out[0].fiscal_year, out[0].fiscal_period) - _qi(f.fiscal_year, f.fiscal_period) != 1:
            break
        out.insert(0, f)
    return out


def _insufficient(why: str, **payload) -> EvidenceItem:
    return EvidenceItem(agent="forecast", stance="insufficient", summary=why, payload=payload)


def run(s: Session, claim: Claim, ctx: Ctx, lock: Lock) -> tuple[EvidenceItem, Usage]:
    ty, tp = claim.target_fiscal_year, claim.target_fiscal_period
    if ty is None or tp is None or tp == "FY":
        # ponytail: fiscal-year guidance would need summed quarterly forecasts with a joint
        # interval; out of scope, so it is left to the other agents.
        return _insufficient("forecast covers quarterly guidance only"), OFFLINE
    with lock:
        series = get_series(s, ctx.cik, claim.metric, "Q")
    known = period_key(claim.fiscal_year, claim.fiscal_period)
    hist = contiguous_tail([f for f in series
                            if period_key(f.fiscal_year, f.fiscal_period) <= known])
    return assess(claim, hist), OFFLINE


def assess(claim: Claim, hist: list[Fact]) -> EvidenceItem:
    """Pure check of quarterly guidance against a contiguous quarterly history."""
    if not hist:
        return _insufficient("no quarterly XBRL history for this metric")
    h = _qi(claim.target_fiscal_year, claim.target_fiscal_period) - _qi(
        hist[-1].fiscal_year, hist[-1].fiscal_period)
    if not 1 <= h <= MAX_H:
        return _insufficient(f"guidance horizon of {h} quarters is outside 1..{MAX_H}")
    y = np.array([f.value for f in hist])
    if len(y) < HOLDOUT + MIN_TRAIN + h - 1:
        return _insufficient("quarterly history too short for a held-out skill check",
                             history_len=len(y))
    unit = CONCEPTS[claim.metric].unit
    if claim.unit not in ((unit,) if claim.comparison is None else ("percent", unit)):
        return _insufficient(f"claim unit {claim.unit} does not match {claim.metric}")
    lo = float(claim.value)
    hi = float(claim.value_high) if claim.value_high is not None else lo
    if claim.comparison:  # growth guidance -> the level it implies, from a filed base quarter
        back = SEASON if claim.comparison == "yoy" else 1
        if back < h:
            return _insufficient("base quarter of the guided change is not yet reported")
        base = y[len(y) - 1 - (back - h)]
        pct = claim.unit == "percent" and unit != "percent"
        lo, hi = ((base * (1 + v / 100) if pct else base + v) for v in (lo, hi))
    skill = backtest(y, h)
    f = forecast_all(y, h)["ensemble"]
    point, flo, fhi = (float(a[-1]) for a in f)
    ref = f"{claim.metric}:{claim.target_fiscal_year}{claim.target_fiscal_period}:ensemble"
    cites = [Citation(kind="forecast", ref_id=f"{ref}:{k}", value=v, unit=unit)
             for k, v in (("point", point), ("lo90", flo), ("hi90", fhi))]
    payload = {"mase": skill, "horizon": h, "history_len": len(y), "point": point,
               "lo90": flo, "hi90": fhi, "guided_low": lo, "guided_high": hi}
    if skill["ensemble"] >= 1.0:
        return EvidenceItem(agent="forecast", stance="insufficient", citations=cites,
                            payload=payload,
                            summary="ensemble does not beat seasonal naive on held-out quarters")
    mid = (lo + hi) / 2
    inside = flo <= mid <= fhi
    return EvidenceItem(
        agent="forecast", stance="supports" if inside else "contradicts", citations=cites,
        payload=payload | {"edge": min(abs(mid - flo), abs(fhi - mid)) / max(fhi - flo, 1e-12)},
        summary=f"Guided value is {'inside' if inside else 'outside'} the 90% interval "
                "implied by quarterly history.",
    )

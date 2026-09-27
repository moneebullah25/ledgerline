"""§10.7 severity: a config-driven weighted sum, weights in ledgerline/severity.toml."""

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "severity.toml"


@dataclass(frozen=True)
class SeverityWeights:
    label: dict[str, float]
    deviation: float
    deviation_cap: float
    unhedged: float
    abnormal_return: float
    abnormal_return_cap: float


@lru_cache
def load_weights(path: Path = DEFAULT_PATH) -> SeverityWeights:
    cfg = tomllib.loads(path.read_text())
    return SeverityWeights(label=cfg["label"], **cfg["weights"])


def severity(
    verdict_label: str,
    confidence: float,
    deviation: float | None,
    hedging_score: float,
    abnormal_return: float | None,
    weights: SeverityWeights,
) -> float:
    """deviation is the relative error |claimed - filed| / |filed| from the XBRL agent;
    abnormal_return is the market agent's event-window abnormal return. Either is None
    when that agent had nothing to say, which contributes 0 rather than a guess."""
    w = weights
    body = (
        confidence
        + w.deviation * min(abs(deviation or 0.0), w.deviation_cap)
        + w.unhedged * (1.0 - hedging_score)
        + w.abnormal_return * min(abs(abnormal_return or 0.0), w.abnormal_return_cap)
    )
    return round(w.label[verdict_label] * body, 4)

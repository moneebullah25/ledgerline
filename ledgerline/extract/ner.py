"""FR-30 entity spans: zero-shot GLiNER for metric names and periods, regex for numbers.

GLiNER is good at noun-phrase boundaries ("Data Center revenue") and bad at turning
"$94.9 billion" into 94.9e9, so money and percentages always come from the regexes in
normalize.py; GLiNER only adds metric and period spans the lexicon/regex miss.
"""

from functools import lru_cache

from ledgerline.config import settings
from ledgerline.extract.normalize import find_periods, match_metrics, parse_values

LABELS = {"financial metric": "metric", "money amount": "money", "percentage": "percent",
          "time period": "period"}
THRESHOLD = 0.5  # GLiNER's documented default; not tuned on the eval set

Span = tuple[int, int, str]  # (start, end, label) with label in LABELS.values()


@lru_cache(maxsize=1)
def _model():
    from gliner import GLiNER  # heavy import (torch); only paid when NER is actually used

    return GLiNER.from_pretrained(settings().ner_model)


def gliner_spans(text: str) -> list[Span]:
    ents = _model().predict_entities(text, list(LABELS), threshold=THRESHOLD)
    return [(e["start"], e["end"], LABELS[e["label"]]) for e in ents]


def regex_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for v in parse_values(text):
        if v.kind == "flat":
            continue
        # Ranges are two entities ("$89 billion" and "$93 billion"); re-parse each side.
        label = "money" if v.unit.startswith("USD") else "percent"
        parts = [p for p in parse_values(text[v.start:v.end]) if p.value_high is None] or [v]
        spans += [(v.start + p.start, v.start + p.end, label) for p in parts]
    spans += [(m.start, m.end, "metric") for m in match_metrics(text)]
    spans += [(s, e, "period") for s, e, *_ in find_periods(text)]
    return spans


def entity_spans(text: str, use_gliner: bool = True) -> list[Span]:
    """Regex spans, plus GLiNER metric/period spans that do not overlap one of them.

    Where GLiNER and the lexicon both find a metric, the GLiNER span wins when it is wider:
    its boundary carries the qualifier ("Data Center revenue") that decides mapping.
    """
    spans = regex_spans(text)
    if not use_gliner:
        return spans
    for s, e, label in gliner_spans(text):
        if label not in ("metric", "period"):
            continue
        overlapping = [x for x in spans if x[2] == label and x[0] < e and s < x[1]]
        if label == "metric" and overlapping and all(s <= x[0] and x[1] <= e for x in overlapping):
            spans = [x for x in spans if x not in overlapping]
            overlapping = []
        if not overlapping:
            spans.append((s, e, label))
    return sorted(spans)

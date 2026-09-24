"""Deterministic parsing: numeric values, fiscal periods, metric names, hedging.

GLiNER (ner.py) only proposes spans. Everything that becomes a stored number or period
goes through the regexes here, so the same text always yields the same claim.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

from ledgerline.concepts import CONCEPTS

# ---------------------------------------------------------------- values


@dataclass
class Value:
    start: int
    end: int
    value: float
    unit: str  # USD | USD/share | percent
    kind: str  # money | per_share | percent | pp (percentage points) | flat
    value_high: float | None = None


_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mm": 1e6,
          "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "t": 1e12}
_NUM = r"\d[\d,]*(?:\.\d+)?"
# Scale words are matched case-insensitively; single-letter abbreviations must follow the
# number directly ("$94.9B") so "$5 M&A" or "$2 T-shirts" never read as a scale.
_MONEY = re.compile(
    rf"(?P<neg>(?<![\w.])-\s?)?\$\s?(?P<paren>\()?(?P<num>{_NUM})\)?"
    rf"(?:\s?(?P<word>trillion|billion|million|thousand)\b|(?P<abbr>bn|mm|[BMKT])\b)?",
    re.IGNORECASE,
)
_MONEY_RANGE = re.compile(rf"\$(?P<lo>{_NUM})\s?(?:-|–|to)\s?(?P<hi>{_NUM})\s?(?P<word>trillion|billion|million)\b",
                          re.IGNORECASE)
_PER_SHARE = re.compile(r"\s?(?:per (?:diluted |basic |common )*share|a share)\b(?P<diluted>)", re.IGNORECASE)
# "10 to 12 percent", "between 10 and 12%": the first bound is printed without its unit.
_PCT_RANGE = re.compile(rf"(?P<lo>{_NUM})\s?(?:to|-|–|and)\s?(?P<hi>{_NUM})\s?(?:%|percent\b)", re.IGNORECASE)
_PERCENT = re.compile(rf"(?P<neg>-)?(?P<num>{_NUM})\s?(?:%|percent\b|per cent\b)", re.IGNORECASE)
_BPS = re.compile(rf"(?P<num>{_NUM})[\s-]?(?:basis[\s-]points?|bps)\b", re.IGNORECASE)
_PP = re.compile(rf"(?P<num>{_NUM})[\s-]?(?:percentage[\s-]points?|points?|pts)\b", re.IGNORECASE)
_FLAT = re.compile(r"\bflat\b", re.IGNORECASE)
_WORDS = {w: i for i, w in enumerate([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"])}
_W = "|".join(_WORDS)
# "seven percent", "one to three percent", "four and a half percent" (P&G, KO spell them out)
_WORD_PCT = re.compile(rf"\b(?P<lo>{_W})(?P<half>\s+and\s+a\s+half)?(?:\s+(?:to|and)\s+(?P<hi>{_W})"
                       rf"(?P<hhalf>\s+and\s+a\s+half)?)?\s+percent\b", re.IGNORECASE)
_RANGE_SEP = re.compile(r"^\s*(?:to|and|through|-|–|—)\s*$", re.IGNORECASE)
_PLUS_MINUS = re.compile(r"^,?\s*(?:plus or minus|\+/-|±)\s*$", re.IGNORECASE)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def scaled(num: str | float, scale: float) -> float:
    """Exact decimal scaling: 133.7 billion is 133700000000.0, not ...99998."""
    return float(Decimal(str(num).replace(",", "")) * Decimal(int(scale)))


def parse_values(text: str) -> list[Value]:
    """Money, per-share, percent, basis-point and percentage-point values, ranges merged.

    "$89 billion to $93 billion" -> one Value(89e9, value_high=93e9); "$89 to $93 billion"
    borrows the scale from the upper bound; "$108.0 billion, plus or minus 2%" becomes the
    range it describes; "flat to 2.0%" is the range 0..2.
    """
    vals: list[Value] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for m in _MONEY_RANGE.finditer(text):  # "$45-48 billion": one "$", one scale word
        scale = _SCALE[m["word"].lower()]
        lo, hi = scaled(m["lo"], scale), scaled(m["hi"], scale)
        if hi >= lo:
            vals.append(Value(m.start(), m.end(), lo, "USD", "money", hi))
            taken.append((m.start(), m.end()))
    for m in _MONEY.finditer(text):
        if not free(m.start(), m.end()):
            continue
        scale_word = (m["word"] or m["abbr"] or "").lower()
        v = scaled(m["num"], _SCALE.get(scale_word, 1.0))
        if m["neg"] or m["paren"]:
            v = -v
        end = m.end()
        kind, unit = "money", "USD"
        ps = _PER_SHARE.match(text, end)
        if ps and not scale_word:
            kind, unit, end = "per_share", "USD/share", ps.end()
        vals.append(Value(m.start(), end, v, unit, kind))
        taken.append((m.start(), end))
    for m in _PCT_RANGE.finditer(text):
        lo, hi = _num(m["lo"]), _num(m["hi"])
        if free(m.start(), m.end()) and hi >= lo and (not m.start() or text[m.start() - 1] not in "$.,"):
            vals.append(Value(m.start(), m.end(), lo, "percent", "percent", hi))
            taken.append((m.start(), m.end()))
    for rx, kind, factor in ((_BPS, "pp", 0.01), (_PP, "pp", 1.0), (_PERCENT, "percent", 1.0)):
        for m in rx.finditer(text):
            if not free(m.start(), m.end()) or (m.start() and text[m.start() - 1] in "$.,"):
                continue
            v = _num(m["num"]) * factor
            if m.groupdict().get("neg"):
                v = -v
            vals.append(Value(m.start(), m.end(), round(v, 6), "percent", kind))
            taken.append((m.start(), m.end()))
    for m in _WORD_PCT.finditer(text):
        lo = _WORDS[m["lo"].lower()] + (0.5 if m["half"] else 0.0)
        hi = _WORDS[m["hi"].lower()] + (0.5 if m["hhalf"] else 0.0) if m["hi"] else None
        if hi is None or hi >= lo:
            vals.append(Value(m.start(), m.end(), lo, "percent", "percent", hi))
    for m in _FLAT.finditer(text):
        vals.append(Value(m.start(), m.end(), 0.0, "percent", "flat"))
    vals.sort(key=lambda v: v.start)
    return _merge_ranges(text, vals)


def _merge_ranges(text: str, vals: list[Value]) -> list[Value]:
    out: list[Value] = []
    i = 0
    while i < len(vals):
        a = vals[i]
        b = vals[i + 1] if i + 1 < len(vals) else None
        gap = text[a.end:b.start] if b else ""
        if b and _PLUS_MINUS.match(gap) and a.kind in ("money", "percent", "per_share"):
            # "$108.0 billion, plus or minus 2%" / "74.0%, plus or minus 50 basis points"
            if b.kind == "percent" and a.kind != "percent":
                lo, hi = a.value * (1 - b.value / 100), a.value * (1 + b.value / 100)
            else:
                lo, hi = a.value - abs(b.value), a.value + abs(b.value)
            out.append(Value(a.start, b.end, round(lo, 6), a.unit, a.kind, round(hi, 6)))
            i += 2
            continue
        # Same kind only: "increased 890 basis points to 49.2 percent" is a change and a level.
        # "and" joins a range only after "between": "74.9% and 75.0%" is two values.
        and_ok = gap.strip() != "and" or text[:a.start].rstrip().lower().endswith("between")
        # "$0.13 to $0.17 per share": the lower bound was printed without "per share"
        kinds_ok = b and (a.kind == b.kind or a.kind == "flat" or (a.kind, b.kind) == ("money", "per_share"))
        if kinds_ok and and_ok and _RANGE_SEP.match(gap):
            lo, hi = a.value, b.value
            if a.kind == "money" and b.kind == "money" and abs(lo) < 1e3 <= abs(hi):
                # "$89 to $93 billion": the lower bound was printed without its scale.
                lo *= _scale_of(hi, lo)
            if hi >= lo:
                out.append(Value(a.start, b.end, lo, b.unit, b.kind, hi))
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def _scale_of(hi: float, lo: float) -> float:
    for s in (1e12, 1e9, 1e6, 1e3):
        if abs(hi) >= s and abs(lo) * s <= abs(hi) * 10:
            return s
    return 1.0


_CELL = re.compile(rf"^(?P<neg>-)?\$?\s?(?P<paren>\()?\$?\s?(?P<num>{_NUM})\)?\s?(?P<pct>%|pts?)?\)?$")


def parse_cell(cell: str) -> tuple[float, str | None] | None:
    """A table cell -> (value, suffix) with suffix '%', 'pts' or None. "(1,234)" is -1234."""
    m = _CELL.match(cell.strip())
    if not m:
        return None
    v = _num(m["num"])
    if m["neg"] or m["paren"]:
        v = -v
    suffix = "pts" if (m["pct"] or "").startswith("pt") else m["pct"]
    return v, suffix


# ---------------------------------------------------------------- periods

_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_YR = r"['’]?(\d{4}|\d{2})\b"
_FISCAL = r"(?:fiscal(?:\s+year)?|fy)\s?"
_PERIOD_RX = [
    # "fiscal 2026 third quarter", "FY26 Q3"
    ("fq", re.compile(rf"\b{_FISCAL}{_YR}\s+(first|second|third|fourth)[\s-]quarter", re.IGNORECASE)),
    # "third quarter of fiscal 2027", "second quarter fiscal 2026", "third-quarter 2026"
    ("qf", re.compile(rf"\b(first|second|third|fourth)[\s-]quarter(?:\s+of)?(?:\s+{_FISCAL}{_YR}|\s+(\d{{4}})\b)?",
                      re.IGNORECASE)),
    # "Q2 FY27", "Q4 FY 2026", "Q3 2026", "Q1"
    ("q", re.compile(rf"\bQ([1-4])\b(?:\s*{_FISCAL}{_YR}|\s+(\d{{4}})\b)?", re.IGNORECASE)),
    ("next", re.compile(r"\bnext quarter\b", re.IGNORECASE)),
    ("thisq", re.compile(r"\b(?:for|in|during) the (?:current )?quarter\b", re.IGNORECASE)),
    # "quarter ended June 27, 2026"; "fiscal year ended June 30, 2026"
    ("qended", re.compile(r"\bquarter ended\s+([A-Z][a-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")),
    ("fyended", re.compile(r"\b(?:fiscal\s+)?year ended\s+([A-Z][a-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")),
    # "December quarter", "June quarter"
    ("mq", re.compile(r"\b(March|June|September|December)\s+quarter\b", re.IGNORECASE)),
    # "full year 2025": one period, not "full year" plus a bare year
    ("fy", re.compile(rf"\bfull[\s-]year\s+(?:{_FISCAL})?(20\d\d)\b", re.IGNORECASE)),
    # "full year fiscal 2027", "fiscal 2025", "fiscal year 2026", "FY 2027", "FY27"
    ("fy", re.compile(rf"\b(?:full[\s-]year\s+)?{_FISCAL}{_YR}", re.IGNORECASE)),
    # "annual revenues", but not "its annual Prime Big Deal Days" (an event)
    ("fullyear", re.compile(r"\b(?:full[\s-]year|fiscal year|for the year|(?<!trailing )twelve months"
                            r"|annual(?!\s+(?-i:[A-Z])))\b", re.IGNORECASE)),
    # bare year ("2026 capital expenditures"): fiscal year for calendar-year filers; dates
    # ("September 30, 2025", "June 2026") are rejected in find_periods
    ("year", re.compile(r"\b(20\d\d)\b")),
]
_DATE_BEFORE = re.compile(r"(?:[A-Z][a-z]+\.?\s+\d{1,2},\s*|[A-Z][a-z]+\.?\s+)$")


def _year(s: str) -> int:
    y = int(s)
    return y + 2000 if y < 100 else y


def _fye_month(fiscal_year_end: str) -> int:
    month, day = int(fiscal_year_end[:2]), int(fiscal_year_end[2:])
    # 52/53-week years ending in the first days of a month really close the prior month.
    return 12 if month == 1 and day <= 7 else (month - 1 if day <= 7 else month)


def fiscal_quarter(month: int, day: int, fiscal_year_end: str) -> int:
    """Fiscal quarter a calendar quarter-end date falls in, e.g. AAPL (0927) December -> 1.
    The fiscal year is not derived here: see resolve_period."""
    if day <= 7:  # a 52/53-week quarter ending Jan 3 really closes December
        month = 12 if month == 1 else month - 1
    return ((month - _fye_month(fiscal_year_end) - 1) % 12) // 3 + 1


def _pidx(fp: str) -> int:
    return 4 if fp == "FY" else int(fp[1])


def _infer_year(q: int, fy: int, fp: str, forward: bool) -> int:
    cur = _pidx(fp)
    if forward:
        return fy + 1 if q <= cur else fy
    return fy - 1 if q > cur else fy


def find_periods(text: str) -> list[tuple[int, int, str, tuple]]:
    """Every period expression: (start, end, kind, groups), overlapping matches removed."""
    hits: list[tuple[int, int, str, tuple]] = []
    for kind, rx in _PERIOD_RX:
        for m in rx.finditer(text):
            if kind == "year" and (_DATE_BEFORE.search(text[:m.start()]) or text[m.end():m.end() + 1] == ","):
                continue
            if all(m.end() <= s or m.start() >= e for s, e, *_ in hits):
                hits.append((m.start(), m.end(), kind, m.groups()))
    return sorted(hits)


def resolve_period(kind: str, g: tuple, fy: int, fp: str, *, forward: bool = False,
                   fiscal_year_end: str | None = None) -> tuple[int, str] | None:
    """One find_periods hit -> (fiscal_year, period) relative to the source quarter (fy, fp)."""
    if kind == "fq":
        return _year(g[0]), f"Q{_ORD[g[1].lower()]}"
    if kind == "qf":
        q = _ORD[g[0].lower()]
        yr = g[1] or g[2]
        return (_year(yr) if yr else _infer_year(q, fy, fp, forward)), f"Q{q}"
    if kind == "q":
        q = int(g[0])
        yr = g[1] or g[2]
        return (_year(yr) if yr else _infer_year(q, fy, fp, forward)), f"Q{q}"
    if kind == "thisq":  # "For the quarter, net revenues grew 2%" under a "Quarterly/Full Year" heading
        return (fy, fp) if fp != "FY" else None
    if kind == "next":
        cur = _pidx(fp)
        return (fy + 1, "Q1") if cur == 4 else (fy, f"Q{cur + 1}")
    if kind in ("qended", "fyended"):
        month = _MONTHS.get(g[0].lower())
        if month is None:
            return None
        if kind == "fyended":
            return fy, "FY"  # the release's own fiscal year; the date is its last day
        if fiscal_year_end is None:
            return None
        # Only the quarter number comes from the date: companies name fiscal years by the
        # calendar year they end (NVDA) or start (HD), so the year is taken relative to
        # the source quarter, exactly as for "fourth quarter".
        q = fiscal_quarter(month, int(g[1]), fiscal_year_end)
        return _infer_year(q, fy, fp, forward), f"Q{q}"
    if kind == "mq":
        if fiscal_year_end is None:
            return None
        q = fiscal_quarter(_MONTHS[g[0].lower()], 28, fiscal_year_end)
        return _infer_year(q, fy, fp, forward), f"Q{q}"
    if kind in ("fy", "year"):
        return _year(g[0]), "FY"
    if kind == "fullyear":
        return fy, "FY"
    return None


# ---------------------------------------------------------------- metrics

# Phrases seen in releases that CONCEPTS does not list; kept here rather than widening the
# shared vocabulary, since each one is only safe with the qualifier checks below.
_EXTRA = {
    "earnings per share": "eps_diluted",
    "diluted earnings-per-share": "eps_diluted",
    "diluted net income per share": "eps_diluted",
    "net income per diluted share": "eps_diluted",
    "cash flows from operations": "operating_cash_flow",
    "cash flow from operating activities": "operating_cash_flow",
    "net cash from operations": "operating_cash_flow",
    "diluted net earnings per share": "eps_diluted",
    "net earnings per diluted share": "eps_diluted",
    "gross margins": "gross_margin",
    "operating margins": "operating_margin",
}
SYNONYMS: dict[str, str] = {syn: c.name for c in CONCEPTS.values() for syn in c.synonyms} | _EXTRA
_SYN_RX = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(s) for s in sorted(SYNONYMS, key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE,
)

# Words that may precede a metric without changing what it measures. Any other word right
# before a metric name is read as a qualifier ("Data Center revenue", "memory revenue"): a
# missed claim is cheaper than a segment figure checked against a company-wide fact.
_NEUTRAL = {
    "total", "net", "gaap", "consolidated", "company", "company's", "quarterly",
    "record", "diluted", "fiscal", "full", "year", "full-year", "annual", "all-time", "overall",
    "strong", "higher", "lower", "first", "second", "third", "fourth", "quarter", "current",
    "prior", "sequential", "all-in",
}
_VERB = re.compile(r"^(?:expect|anticipat|project|forecast|estimat|deliver|report|generat|post"
                   r"|achiev|record|see|saw|had|grew|grow|increas|decreas|reaffirm|rais|guid|target)\w*$")
_OPENERS = set("“\"•◦▪:")
_STOP = {"was", "were", "is", "are", "be", "and", "or", "of", "for", "in", "the", "with", "on",
         "as", "by", "to", "from", "a", "an", "its", "our", "their", "that", "which", "while",
         "whose", "both", "each", "than", "versus", "vs", "said", "today", "we", "i", "they",
         "it", "you"}
# A basis other than the one XBRL reports: verifying these against GAAP facts is invalid.
NON_GAAP = re.compile(
    r"non-gaap|\badjusted\b|\bcomparable\b|\borganic\b|constant currenc(?:y|ies)|currency[\s-]neutral"
    r"|\bexcluding\b|\bpro forma\b",
    re.IGNORECASE,
)
_PERIODISH = re.compile(r"^(?:(?:first|second|third|fourth)-quarter|q[1-4]|fy\d*|\d{4}|full-year)$", re.IGNORECASE)
_TOKEN = re.compile(r"[\w&'’()-]+|[^\w\s]")


@dataclass
class Mention:
    start: int
    end: int
    metric: str | None  # None: a metric the vocabulary does not cover, or a qualified one
    non_gaap: bool = False
    loss: bool = False  # "net loss", "loss from operations": the value is the negative of the concept


def qualifier_start(prefix: str) -> int | None:
    """Offset in `prefix` where the words that narrow the following metric name to a slice of
    the company begin ("Data Center " -> 0), or None when nothing narrows it."""
    toks = list(_TOKEN.finditer(prefix))
    verb = False
    for i in range(len(toks) - 1, max(-1, len(toks) - 6), -1):
        tok = re.sub(r"\(\d+\)", "", toks[i][0])
        t = tok.lower().strip("()")
        if not re.match(r"\w", tok) or t in _STOP:
            return None
        if _VERB.match(t):
            verb = True
            continue
        if t in _NEUTRAL or _PERIODISH.match(t) or NON_GAAP.fullmatch(t):
            continue
        # A capitalised first word before a verb is the subject: "Adobe achieved record revenue".
        initial = i == 0 or toks[i - 1][0] in _OPENERS
        if initial and verb and tok[:1].isupper():
            return None
        # the whole qualifier: "Data" as well as "Center", "Google Cloud" as well as "Cloud"
        while i > 0 and re.match(r"\w", toks[i - 1][0]) and not _is_plain(toks[i - 1][0]):
            i -= 1
        return toks[i].start()
    return None


def _is_plain(tok: str) -> bool:
    t = tok.lower()
    return t in _STOP or t in _NEUTRAL or bool(_VERB.match(t) or _PERIODISH.match(t) or NON_GAAP.fullmatch(t))


def match_metrics(sentence: str) -> list[Mention]:
    """Vocabulary hits with the qualifier checks that keep "Data Center revenue" and
    "non-GAAP EPS" from being read as company-wide GAAP revenue and EPS."""
    out = []
    for m in _SYN_RX.finditer(sentence):
        metric: str | None = SYNONYMS[m[1].lower()]
        prefix, after = sentence[:m.start()], sentence[m.end():m.end() + 40]
        start = m.start()
        if (q := qualifier_start(prefix)) is not None:
            metric, start = None, q  # the claim text keeps "Data Center" in front of "revenue"
        loss = "loss" in m[1].lower()
        if loss and re.match(r"\s+(?:on|from|related|associated|attributable to non)|\s*carry", after, re.IGNORECASE):
            # "net loss on investments", "net operating loss carryforwards": not the concept, so
            # the printed value is kept as printed
            metric, loss = None, False
        if re.match(r"\s+(?:in|from|for)\s+(?!(?i:the|fiscal|fy|full|first|second|third|fourth|quarter|year)\b"
                    r"|Q[1-4]\b)[A-Z]", after):
            metric = None  # "Revenue in Intelligent Cloud", "Revenues for Converse"
        acr = re.match(r"\s+and\s+([A-Z]{2,})\b", after)
        if acr and acr[1].lower() not in SYNONYMS and acr[1] != "GAAP":
            metric = None  # "Revenue and RPO Surpasses $20 Billion": a sum with a non-GAAP quantity
        if metric == "net_income" and re.match(r"\s+per\b", after):
            metric = None  # "net earnings per share": basic EPS, not net income
        if re.match(r"\s*(?:run[\s-]rate|backlog|mix|per\s+(?!share|diluted))", after, re.IGNORECASE):
            metric = None  # "revenue run-rate", "revenue per employee": other quantities
        window = " ".join(_TOKEN.findall(prefix)[-4:]) + " " + after[:14]
        non_gaap = bool(NON_GAAP.search(window)) and not re.search(r"gaap and non-gaap", window, re.IGNORECASE)
        out.append(Mention(start, m.end(), metric, non_gaap, loss))
    return out


# ---------------------------------------------------------------- hedging

_HEDGES = re.compile(
    r"\b(?:may|might|could|would|should|approximately|approx\.|about|around|roughly|nearly|"
    r"almost|expects?|expected|expecting|anticipates?|anticipated|believes?|estimates?|estimated|"
    r"likely|potential(?:ly)?|projects?|projected|forecast(?:s|ing)?|targets?|aims?|intends?|"
    r"plans?|outlook|guidance|subject to|plus or minus|up to|in the range of|between|assumes?|"
    r"assuming|preliminary)\b|~|±",
    re.IGNORECASE,
)


def hedging_score(text: str) -> float:
    """h = min(1, hits / (1 + words / 10)).

    hits = hedge-lexicon matches in the claim's clause. Dividing by (1 + words/10) keeps one
    "approximately" in a long sentence from scoring like "is expected to be about" in a short
    one: a 10-word clause with one hedge scores 0.5, with two scores 1.0; a 30-word clause
    with one hedge scores 0.25. No hedge -> 0.
    """
    hits = len(_HEDGES.findall(text))
    words = len(text.split())
    return round(min(1.0, hits / (1 + words / 10)), 3) if hits else 0.0

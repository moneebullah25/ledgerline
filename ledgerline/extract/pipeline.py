"""S4 claim extraction: text -> typed, normalized, span-anchored ClaimDraft rows.

Rules first (FR-31), then the FR-32 LLM normalization pass (extract/refine.py), which may only
null or re-map a claim's metric and target period and is a no-op offline. Per line of the
source text:

  table row ("Revenue | $96,221 | ...")  -> reported/comparison claims, only when the table
                                            header states both the scale and the period
  table row under an outlook heading      -> guidance, from the GAAP column, when the target
                                            period lies after the reported quarter
  safe-harbor / boilerplate              -> dropped before any parsing
  heading or lead-in ("Outlook", "...as follows:") -> section context (guidance?, period)
  prose sentence                         -> one claim per value, typed by local cues;
                                            no value but evaluative language -> qualitative

Unmapped metrics: a value whose metric is outside CONCEPTS, or narrowed to a segment
("Data Center revenue"), is kept as claim_type "qualitative" with metric None and its value,
so the filing-RAG agent can still check it. Non-GAAP and constant-currency values are
dropped: no SEC filing states them as reported facts, so they can only ever be unverifiable.
So are trailing-twelve-month and year-to-date values, which have no fiscal period here.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ledgerline import llm
from ledgerline.concepts import CONCEPTS
from ledgerline.extract.normalize import (
    NON_GAAP,
    SYNONYMS,
    Mention,
    Value,
    find_periods,
    hedging_score,
    match_metrics,
    parse_cell,
    parse_values,
    qualifier_start,
    resolve_period,
    scaled,
)
from ledgerline.models import CallAudio, Claim, Company, Filing, Segment
from ledgerline.schemas import ClaimDraft

# Rows record which pass produced them: rules plus the LLM normalization pass, or rules alone
# (offline). Callers look up claims at extractor_version(), so claims extracted offline are
# re-extracted once credentials appear.
EXTRACTOR_VERSION = "s4-llm-1"
RULES_VERSION = "s4-rules-1"


def extractor_version() -> str:
    """The version extraction writes now: the LLM pass runs exactly when llm.available()."""
    return EXTRACTOR_VERSION if llm.available() else RULES_VERSION


log = logging.getLogger(__name__)

# ---------------------------------------------------------------- cues

_STRONG_BOILER = re.compile(
    r"forward-looking statements?|safe harbor|private securities litigation|cautionary (?:note|statement)"
    r"|risk factors|actual results (?:may|could) differ",
    re.IGNORECASE,
)
_BOILER = re.compile(
    r"conference call|webcast|replay|investor relations|press release|all rights reserved|trademark"
    r"|reconcil|non-gaap (?:financial )?measures?\b|@|https?://|www\.|\(\d{3}\)\s?\d{3}-\d{4}"
    r"|\d{3}-\d{3}-\d{4}|for immediate release|contacts?:|note to editors|subject to board"
    r"|shareholders of record|stockholders of record|is available at|will be available"
    r"|comparisons? relates?|otherwise noted|comparative percentages",
    re.IGNORECASE,
)
_FORWARD = re.compile(
    r"\b(?:expects?|expected (?:to|in)|is expected|are expected|expecting|anticipates?|forecast(?:s|ing)?"
    r"|outlook (?:for|of|is)|guidance (?:for|of|is|assumes)|estimates? that|projects?|projected|will be"
    r"|to be (?:between|approximately|about|in the range)|plans? to|intends? to|to grow|to increase"
    r"|to decline)\b|(?<![A-Za-z])targets? (?:assumes?|of|for)\b",
    re.IGNORECASE,
)
_PAST = re.compile(r"\b(?:was|were|reports|reported|grew|increased|decreased|declined|rose|fell|delivered"
                   r"|achieved|generated|posted|returned|had|exceeded|surpass(?:es|ed))\b", re.IGNORECASE)
_CHANGE_BEFORE = re.compile(
    r"\b(?P<cue>up|down|(?:in|de)creases? of|(?:in|de)crease[sd]?|grew|grow|growth(?: of)?"
    r"|rose|fell|declined?|decline of|expanded|contracted|improved|higher|lower)"
    r"(?:\s+(?:by|of|approximately|about|nearly|more than|over|roughly|to|be|between|in the range of"
    r"|over (?:the )?(?:prior|previous) (?:year|quarter)))*\s*$",
    re.IGNORECASE,
)
_CHANGE_AFTER = re.compile(r"^\s*(?:year[\s-]over[\s-]year\s+|yoy\s+|sequential\s+)?"
                           r"(?P<cue>growth|increase|decrease|decline|improvement)\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"down|decrease|decline|fell|lower|contracted", re.IGNORECASE)
_QOQ = re.compile(r"previous quarter|prior quarter|sequential|quarter[\s-]over[\s-]quarter|\bq/q\b"
                  r"|last quarter", re.IGNORECASE)
_YOY = re.compile(r"year[\s-]over[\s-]year|a year ago|\byoy\b|\by/y\b|prior year|last year|same period",
                  re.IGNORECASE)
_BASELINE = re.compile(r"\b(?:compared (?:with|to)|versus|vs\.?)\s", re.IGNORECASE)
_BASELINE_END = re.compile(r";|,\s+and\b")
# "versus the prior year to $1.63": the baseline ends where the current value starts
_ZONE_END = re.compile(_BASELINE_END.pattern + r"|\s+to\s+(?=[$\d])")
_RELATIVE = re.compile(r"(?:last|prior|previous|same period of|a year ago|corresponding period of)\s*$",
                       re.IGNORECASE)
_DROP_AFTER = re.compile(r"^[^,;]{0,25}?(?:constant currenc(?:y|ies)|currency[\s-]neutral)"
                         r"|^\s*(?:benefit|charge|impact|headwind|tailwind)", re.IGNORECASE)
# A component of the number, not the number: "including a favorable impact of $0.11".
_COMPONENT = re.compile(r"(?:\b(?:includ(?:e|es|ing|ed)|impact|benefit|charge|effect|contribut\w*)\b[^,;$%]*"
                        r"|\bfrom\s*"
                        r"|\bprior (?:range|outlook|guidance|expectations?)\b[^,;$%]*)$", re.IGNORECASE)
# A sentence led by "Excluding ..." or "Without ..." restates figures on another basis.
_EX_BASIS = re.compile(r"(?:excluding|without)\b", re.IGNORECASE)
_TTM = re.compile(r"trailing[\s-]twelve[\s-]months?|\bttm\b", re.IGNORECASE)
_RUN_SEP = re.compile(r"\s*(?:,\s*and|,|and)\s*")
_MAX_GAP = 100  # chars between a metric name (or its previous value) and a value it owns
_QUAL_CUE = re.compile(
    r"\b(?:record|strong(?:est)?|exceptional|exceed(?:ed|ing)?|beat|double-digit|all-time high"
    r"|accelerat\w*|momentum|robust|solid|in line with (?:our )?expectations|headwinds?|challeng\w+"
    r"|improv\w+|grew|growth|declin\w+|weak\w*)\b",
    re.IGNORECASE,
)
_QUAL_SUBJECT = re.compile(r"\b(?:results|quarter|demand|performance|business|margins?"
                           r"|profitability|earnings)\b", re.IGNORECASE)
_NEUTRAL_HEADING = re.compile(
    r"results|highlights|summary|financial|outlook|guidance|targets?|quarter|fiscal|year|review"
    r"|performance|overview|income|statements?|consolidated|expects?", re.IGNORECASE)
_SKIP_HEADING = re.compile(r"(?<!gaap and )non-gaap|definition|forward-looking|reconciliation|safe harbor"
                           r"|about [A-Z]|cautionary", re.IGNORECASE)
_YTD = re.compile(r"year[\s-]to[\s-]date|\b(?:six|nine) months\b", re.IGNORECASE)
_BULLET = "•◦▪·-–*"
_ABBREV = {"inc", "corp", "co", "ltd", "u.s", "no", "vs", "st", "jr", "mr", "ms", "dr", "p.m",
           "a.m", "calif", "wash", "ore", "approx", "e.g", "i.e", "n.a", "u.k", "dept"}

# ---------------------------------------------------------------- helpers


def _label_row(line: str) -> bool:
    """"Cost of sales:" above its table rows."""
    return line.endswith(":") and len(line) < 60


def _key(fy: int, fp: str) -> tuple[int, int]:
    return fy, 4 if fp == "FY" else int(fp[1])


def _sentences(line: str) -> list[tuple[int, int]]:
    """Sentence offsets inside one line, leading bullets and trailing space excluded."""
    out, start = [], 0
    for m in re.finditer(r"[.!?][\"”’]?\s+(?=[\"“‘A-Z•$])", line):
        word = re.findall(r"[\w.]+$", line[start:m.start()])
        if word and word[0].lower().rstrip(".") in _ABBREV:
            continue
        out.append((start, m.end()))
        start = m.end()
    out.append((start, len(line)))
    spans = []
    for s, e in out:
        while s < e and (line[s].isspace() or line[s] in _BULLET):
            s += 1
        while e > s and line[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))
    return spans


@dataclass
class _Ctx:
    """What the most recent heading or lead-in line says about the lines under it."""
    guidance: bool = False
    period: tuple[int, str] | None = None
    segment: bool = False  # heading names a business segment: no company-wide metric below
    skip: bool = False  # safe harbor, non-GAAP definitions: nothing below is a claim
    last_metric: Mention | None = None
    heading: str = ""


def _heading_ctx(line: str, fy: int, fp: str, fye: str | None) -> _Ctx | None:
    s = line.strip()
    if not s:
        return None
    lead_in = s.endswith(":")
    if parse_values(s) or not (lead_in or (len(s) < 90 and not s.endswith(".") and s[0] not in _BULLET)):
        return None
    guidance = bool(re.search(r"\b(?:outlook|guidance|targets?|expects? to|forecast)\b", s, re.IGNORECASE))
    period = None
    for a, _, kind, g in find_periods(s):
        if not _RELATIVE.search(s[:a]):
            period = resolve_period(kind, g, fy, fp, forward=guidance, fiscal_year_end=fye)
            break
    # "Year-to-Date Results", "Nine Months Ended ...": cumulative figures, no fiscal period of ours
    ytd = bool(_YTD.search(s)) and not re.search(r"quarter", s, re.IGNORECASE)
    return _Ctx(guidance, period, segment=not _NEUTRAL_HEADING.search(s),
                skip=bool(_SKIP_HEADING.search(s)) or ytd, heading=s)


def _mentions(sent: str, use_ner: bool) -> list[Mention]:
    ms = match_metrics(sent)
    if not use_ner:
        return ms
    from ledgerline.extract.ner import gliner_spans

    for gs, ge, label in gliner_spans(sent):
        if label != "metric":
            continue
        inside = [m for m in ms if gs <= m.start and m.end <= ge]
        for m in inside:
            if m.metric and (q := qualifier_start(sent[gs:m.start])) is not None:
                m.metric, m.start = None, gs + q  # GLiNER's wider span carries a segment qualifier
        if not inside and not any(m.start < ge and gs < m.end for m in ms):
            ms.append(Mention(gs, ge, None, bool(NON_GAAP.search(sent[max(0, gs - 20):ge]))))
    return sorted(ms, key=lambda m: m.start)


def _as_loss(value: float, value_high: float | None) -> tuple[float, float | None]:
    """A loss of $X is the signed concept at -X: "net loss of $1.2 billion" -> -1.2e9 and
    "$(0.66)" stays -0.66; a range keeps low <= high."""
    if value_high is None:
        return -abs(value), None
    return -max(abs(value), abs(value_high)), -min(abs(value), abs(value_high))


def _baseline_zones(sent: str) -> list[tuple[int, int]]:
    zones = []
    for m in _BASELINE.finditer(sent):
        end = _ZONE_END.search(sent, m.end())
        zones.append((m.start(), end.start() if end else len(sent)))
    return zones


def _align_unit(metric: str | None, v: Value, is_change: bool) -> tuple[str | None, str]:
    """Reconcile the metric with the value's unit; None metric when they cannot agree."""
    if metric is None:
        return None, v.unit
    want = CONCEPTS[metric].unit
    if is_change:
        return metric, "percent"
    if v.unit == want:
        return metric, want
    if want == "USD/share" and v.kind == "money" and abs(v.value) < 1000:
        return metric, "USD/share"  # "EPS was $0.97"
    if metric == "gross_margin" and v.unit == "USD":
        return "gross_profit", "USD"  # Apple labels gross profit dollars "gross margin"
    return None, v.unit


# ---------------------------------------------------------------- prose


def _periods_named(text: str, fy: int, fp: str, fye: str | None) -> list[tuple[int, str] | None] | None:
    """Periods in reading order, None standing for a year-to-date span (no period of ours).
    Returns None when a period expression cannot be resolved: the order is then unknown."""
    hits = [(a, resolve_period(k, g, fy, fp, fiscal_year_end=fye)) for a, _, k, g in find_periods(text)]
    if any(p is None for _, p in hits):
        return None
    hits += [(m.start(), None) for m in _YTD.finditer(text)]
    return [p for _, p in sorted(hits, key=lambda h: h[0])]


def _respectively(sent: str, values: list[Value], fy: int, fp: str, fye: str | None,
                  heading: str) -> dict[int, tuple[int, str] | None]:
    """Value index -> period for "$59.89 billion and $200.97 billion ... for the fourth quarter
    and full year 2025, respectively": the k-th value of a run takes the k-th period named, in
    the sentence or else in the section heading ("Second-Quarter and Year-to-Date Results").
    None: a year-to-date value, to be dropped."""
    out: dict[int, tuple[int, str] | None] = {}
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1].kind == values[i].kind and _RUN_SEP.fullmatch(
                sent[values[j].end:values[j + 1].start]):
            j += 1
        resp = sent.find("respectively", values[j].end)
        # "GAAP and non-GAAP EPS were $4.90 and $4.77, respectively" pairs bases, not periods
        if j > i and resp != -1 and not re.search(r"gaap and non-gaap", sent, re.IGNORECASE):
            # periods are named after the last value before "respectively", shared by every run
            last = max(v.end for v in values if v.start < resp)
            periods = _periods_named(sent[last:resp], fy, fp, fye)
            if periods == [] and not find_periods(sent):  # "... increased 137% and 72%, respectively"
                periods = _periods_named(heading, fy, fp, fye)
            if periods and len(periods) == j - i + 1:
                out |= dict(zip(range(i, j + 1), periods, strict=True))
        i = j + 1
    return out


def _sentence_claims(sent: str, base: int, ctx: _Ctx, fy: int, fp: str, fye: str | None,
                     use_ner: bool) -> list[ClaimDraft]:
    if _BOILER.search(sent) or _EX_BASIS.match(sent):
        return []
    values = parse_values(sent)
    mentions = _mentions(sent, use_ner) if (values or _QUAL_CUE.search(sent)) else []
    if ctx.segment:
        for m in mentions:
            m.metric = None
    zones = _baseline_zones(sent)
    if any(not any(a <= m.start() < b for a, b in zones) for m in _TTM.finditer(sent)):
        return []  # trailing twelve months: not a fiscal period any fact is filed for
    resp = _respectively(sent, values, fy, fp, fye, ctx.heading)
    # "FY ’25: Net Sales 0%; Organic Sales +2%; Diluted EPS +8%": the label dates every clause
    label = re.match(r"[^:;.]{2,20}:", sent)
    label_hit = next(((k, g) for _, _, k, g in find_periods(label[0])), None) if label else None
    drafts: list[ClaimDraft] = []
    prev_end = 0
    last_end: dict[int, int] = {}
    components: set[int] = set()  # value indices that are parts of another number
    change_cue: dict[int, re.Match | None] = {}  # value index -> the cue that made it a change
    for i, v in enumerate(values):
        nxt = values[i + 1].start if i + 1 < len(values) else len(sent)
        post_end = min(nxt, (re.search(r"[,;.](?:\s|$)", sent[v.end:]) or re.search("$", sent[v.end:])).start()
                       + v.end)
        before, after = sent[prev_end:v.start], sent[v.end:nxt]
        prev_end = v.end
        # "included 6% growth in price/mix and a 1% decline in ...": both are parts of the change
        if _COMPONENT.search(before) or (i - 1 in components and re.fullmatch(r"[^;.]{0,40}\band\b[^;.]*", before)):
            components.add(i)
            continue
        if any(a <= v.start < b for a, b in zones):
            continue  # prior-period baseline, not a claim about this period
        if _DROP_AFTER.search(after) or re.search(r"\bwould have been\s*$", before):
            continue  # constant currency, a component, or a counterfactual
        cue_before = _CHANGE_BEFORE.search(before[-40:])
        if i - 1 in change_cue and values[i - 1].kind == v.kind and _RUN_SEP.fullmatch(before):
            cue_before = change_cue[i - 1]  # "increases of 24% and 22%": the run shares its cue
            change = True
        else:
            cue_after = _CHANGE_AFTER.search(after)
            signed = re.search(r"(?:^|[\s,(])\+\s*$", before)  # "Diluted EPS $1.88, +34%"
            change = v.kind in ("percent", "pp", "flat") and bool(v.kind == "flat" or cue_before or cue_after
                                                                  or signed)
            if change and not cue_before and cue_after:
                cue_before = cue_after  # "a two percent decrease versus the prior year"
        if change:
            change_cue[i] = cue_before
        if v.kind in ("money", "per_share") and _CHANGE_BEFORE.search(before[-25:]) and not re.search(
                r"\bto\s*$", before):
            continue  # dollar deltas ("an increase of $2.6 billion"); "increased to $68.6 billion" is a level
        # metric: "$4 billion of revenue" -> the one right after; else the nearest before
        mention = next((m for m in mentions if m.start >= v.end
                        and re.fullmatch(r"\s*(?:of|in)\s+(?:[\w-]+\s+){0,2}", sent[v.end:m.start])), None)
        if mention is None:
            mention = next((m for m in reversed(mentions) if m.end <= v.start), None)
            # Too far from its metric (or from that metric's previous value): a different subject.
            if mention and v.start - max(mention.end, last_end.get(id(mention), 0)) > _MAX_GAP:
                mention = None
        if mention is None and ctx.last_metric and re.match(r"(?:increase|decrease|up|down)\b", sent, re.IGNORECASE):
            # sub-bullet "Increase of 18% year over year" under "Revenue: ...": the metric carries
            # over, its offsets (from the previous line) do not
            mention = Mention(0, 0, ctx.last_metric.metric, ctx.last_metric.non_gaap)
        if mention is None:
            continue
        between = sent[mention.end:v.start] if mention.end <= v.start else ""
        if re.search(r"would have|constant currenc(?:y|ies)|currency[\s-]neutral", between, re.IGNORECASE):
            continue
        if id(mention) not in last_end and re.search(r",\s+(?!includ|exclud)\w+ing\s", between):
            continue  # "record sales, ..., saving Prime members $1 billion": a clause with its own subject
        last_end[id(mention)] = v.end
        # after the value, only its own words: "$0.31 and non-GAAP EPS of $0.38" does not make $0.31 non-GAAP
        ng_window = (sent[mention.end if mention.end <= v.start else v.end:v.start]
                     + re.split(r"\s+and\s+", sent[v.end:post_end])[0])
        clause_end = _BASELINE_END.search(sent, v.end)
        basis = re.search(r"on an? (non-gaap|gaap|adjusted) basis", sent[v.end:clause_end.start() if clause_end
                                                                     else len(sent)], re.IGNORECASE)
        non_gaap = (mention.non_gaap or re.match(r"\s*on a non-gaap basis", sent, re.IGNORECASE)
                    or bool(re.search(r"non-gaap|\badjusted\b", ng_window, re.IGNORECASE))
                    or bool(basis and basis[1].lower() != "gaap"))
        if re.search(r"gaap and non-gaap", sent[:v.start], re.IGNORECASE):
            # "GAAP and non-GAAP EPS were $2.46 and $2.22, respectively": only the first is GAAP
            first = not any(mention.end < w.start < v.start for w in values)
            non_gaap = "respectively" in sent and not first
        if non_gaap:
            continue
        if mention in mentions:
            ctx.last_metric = mention
        per_diluted = v.kind == "per_share" and "diluted" in sent[v.start:v.end].lower()
        metric = "eps_diluted" if per_diluted else mention.metric
        if v.kind == "per_share" and metric not in (None, "eps_diluted"):
            continue  # "$0.25 per share" next to "net income": a dividend or basic EPS
        metric, unit = _align_unit(metric, v, change)
        if metric is None and mention.metric is not None:
            continue  # the nearest metric name cannot carry this unit: a mis-association

        # typing
        clause_start = max(sent.rfind(";", 0, v.start) + 1, 0)
        clause = sent[clause_start:post_end]
        explicit = None
        for a, b, kind, g in find_periods(sent[clause_start:post_end]):
            a += clause_start
            if any(za <= a < zb for za, zb in zones) or _RELATIVE.search(sent[max(0, a - 25):a]):
                continue
            explicit = (kind, g)
            if a >= v.start:
                break
        forward = bool(_FORWARD.search(clause)) or (ctx.guidance and not _PAST.search(clause))
        target = None
        if explicit:
            target = resolve_period(*explicit, fy, fp, forward=forward, fiscal_year_end=fye)
        if target is None and label_hit:
            target = resolve_period(*label_hit, fy, fp, forward=forward, fiscal_year_end=fye)
        if target is None and ctx.period and ctx.guidance == forward:
            target = ctx.period
        if target is None and not forward:
            target = (fy, fp)
        if i in resp:
            if resp[i] is None:
                continue  # the year-to-date value of a "quarter and year-to-date" pair
            target = resp[i]
        if target and not change and _key(*target) > _key(fy, fp):
            forward = True
        claim_type = "guidance" if forward else "comparison" if change else "reported_result"
        comparison = None
        if change:
            comparison = "qoq" if _QOQ.search(after) and not _YOY.search(after[:_QOQ.search(after).start()]) else "yoy"
        value, value_high = v.value, v.value_high
        if change and cue_before and _NEGATIVE.match(cue_before["cue"]):
            value, value_high = (-value_high, -value) if value_high is not None else (-value, None)
        elif mention.loss and not change:
            value, value_high = _as_loss(value, value_high)

        # span: from the metric name (if it precedes) to the value, plus the yoy/qoq phrase
        start = min(v.start, mention.start) if mention in mentions else v.start
        end = max(v.end, mention.end) if mention in mentions and mention.start > v.end else v.end
        if change:
            cue = (_QOQ if comparison == "qoq" else _YOY).search(sent, v.end, post_end + 30)
            if cue and cue.start() < nxt:
                end = max(end, cue.end())
        drafts.append(ClaimDraft(
            text=sent[start:end], span_start=base + start, span_end=base + end,
            claim_type=claim_type, metric=metric, value=value, value_high=value_high, unit=unit,
            target_fiscal_year=target[0] if target else None,
            target_fiscal_period=target[1] if target else None,
            comparison=comparison,
            hedging_score=hedging_score(sent[min(clause_start, start):end]),
        ))
    if not values and _QUAL_CUE.search(sent) and len(sent.split()) > 5 and (
            mentions or _QUAL_SUBJECT.search(sent)):
        metric = next((m.metric for m in mentions if m.metric), None)
        forward = bool(_FORWARD.search(sent))
        target = ctx.period or (None if forward else (fy, fp))
        drafts.append(ClaimDraft(
            text=sent, span_start=base, span_end=base + len(sent), claim_type="qualitative",
            metric=metric, target_fiscal_year=target[0] if target else None,
            target_fiscal_period=target[1] if target else None, hedging_score=hedging_score(sent),
        ))
    return drafts


# ---------------------------------------------------------------- tables

_SCALE_RX = re.compile(r"\bin (millions|thousands|billions)\b", re.IGNORECASE)
_SCALES = {"millions": 1e6, "thousands": 1e3, "billions": 1e9}


@dataclass
class _Table:
    scale: float | None = None
    period: tuple[int, str] | None = None
    skip: bool = False
    header: list[str] | None = None
    parent: str = ""
    rows_seen: bool = False
    guidance: tuple[int, str] | None = None  # target period of an outlook table ("Financial Targets")


def _table_header(titles: list[str], header_rows: list[str], fy: int, fp: str) -> _Table:
    """Scale and current-period column from the lines above a table. Anything unclear ->
    skip: a wrong scale or column is a wrong number, which is worse than no claim.

    titles: the short non-table lines just above; header_rows: the table's own header rows.
    """
    t = _Table()
    joined = " \n ".join(titles + header_rows)
    if m := _SCALE_RX.search(joined):
        t.scale = _SCALES[m[1].lower()]
    # Only titles decide this: a GAAP table's header row may name a non-GAAP column.
    if re.search(r"segment|reconciliation|non-gaap|balance sheet", " ".join(titles), re.IGNORECASE):
        t.skip = True
    # Columns named after business units ("CCPG | DCAI | ... | Intersegment Eliminations | Total
    # Consolidated"): the first number is a segment's, not the company's.
    if re.search(r"segment|eliminations|all other|corporate unallocated|total consolidated",
                 " ".join(header_rows), re.IGNORECASE):
        t.skip = True
    first = re.search(r"three months ended|quarters? ended|twelve months ended|(?:fiscal )?year ended|six months ended"
                      r"|nine months ended|\bQ([1-4]) FY ?'?(\d{2,4})\b|\b(first|second|third|fourth)[\s-]quarter\b",
                      joined, re.IGNORECASE)
    if first is None:
        t.skip = True
    elif first[1]:
        yr = int(first[2])
        t.period = (yr + 2000 if yr < 100 else yr, f"Q{first[1]}")
    elif first[3]:  # "Third-Quarter | Year-to-Date": only trusted when it names the source quarter
        q = ("first", "second", "third", "fourth").index(first[3].lower()) + 1
        t.period = (fy, fp) if fp == f"Q{q}" else None
    elif re.match(r"three|quarter", first[0], re.IGNORECASE):
        t.period = (fy, fp) if fp != "FY" else None
    elif re.match(r"twelve|(?:fiscal )?year", first[0], re.IGNORECASE):
        t.period = (fy, "FY")
    # A "% Change" column where the current period's value should be: not a level.
    if any(re.search(r"%|change|percent", r.split(" | ")[0], re.IGNORECASE) for r in header_rows):
        t.skip = True
    # Columns must run newest first, or the "first number" is not the current period.
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", joined[first.end():] if first else "")]
    if t.period is None or (years and years[0] < max(years)):
        t.skip = True
    return t


def _has_value(cell: str) -> bool:
    return (parse_cell(cell) is not None and not re.fullmatch(r"(?:19|20)\d\d", cell.strip())) or bool(
        parse_values(cell))


def _guidance_period(titles: list[str], header_rows: list[str], ctx: _Ctx, fy: int, fp: str,
                     fye: str | None) -> tuple[int, str] | None:
    """Target period of a table under an outlook heading: the header's own ("Q3 2026 | GAAP |
    Non-GAAP"), else the lead-in's. None (not an outlook table) unless it lies after (fy, fp)."""
    if re.search(r"reconciliation", " ".join(titles), re.IGNORECASE):
        return None
    hits = find_periods(" ".join(header_rows))
    if any((p := resolve_period(k, g, fy, fp, fiscal_year_end=fye)) and _key(*p) <= _key(fy, fp)
           for _, _, k, g in hits):
        return None  # a column for the reported quarter or earlier: a results table
    period = ctx.period
    if hits:
        period = resolve_period(hits[0][2], hits[0][3], fy, fp, forward=True, fiscal_year_end=fye)
    return period if period and _key(*period) > _key(fy, fp) else None


def _guidance_row(line: str, base: int, t: _Table) -> list[ClaimDraft]:
    """"Total revenue | $6.075 billion to $6.125 billion", "Earnings per share | GAAP: $4.27 to
    $4.32 | Non-GAAP: ..." -> one guidance claim from the GAAP (or only) column."""
    cells = line.split(" | ")
    label = _row_label(cells[0], t.parent)
    if label is None:
        return []
    metric, loss = label
    header = t.header if t.header and len(t.header) == len(cells) else [""] * len(cells)
    offset = len(cells[0]) + 3
    for cell, head in zip(cells[1:], header[1:], strict=True):
        vals = parse_values(cell)
        if vals and not NON_GAAP.search(cell) and not NON_GAAP.search(head):
            break
        offset += len(cell) + 3
    else:
        return []
    v = vals[0]
    change = v.kind in ("percent", "pp") and CONCEPTS[metric].unit != "percent"
    metric, unit = _align_unit(metric, v, change)
    if metric is None:
        return []
    value, value_high = v.value, v.value_high
    scale_word = re.search(r"(?:thousand|million|billion|trillion|\d\s?[BMKT])\b", cell, re.IGNORECASE)
    if unit == "USD" and not scale_word:
        if t.scale is None:
            return []  # a bare "$9.2" with no stated scale is not a number we can store
        value = scaled(value, t.scale)
        value_high = scaled(value_high, t.scale) if value_high is not None else None
    if loss and not change:
        value, value_high = _as_loss(value, value_high)
    end = offset + v.end
    return [ClaimDraft(text=line[:end], span_start=base, span_end=base + end, claim_type="guidance",
                       metric=metric, value=value, value_high=value_high, unit=unit,
                       target_fiscal_year=t.guidance[0], target_fiscal_period=t.guidance[1],
                       comparison="yoy" if change else None, hedging_score=hedging_score(line[:end]))]


def _row_label(label: str, parent: str) -> tuple[str, bool] | None:
    """(metric, is_loss) for a table row label; "Net loss" is net_income with its sign flipped."""
    def norm(s: str) -> str:
        s = re.sub(r"\([^)]*\)|\*|\d+$|\battributable to (?!non)[\w.&]+", "", s, flags=re.IGNORECASE)
        s = s.strip().rstrip(":").strip().lower()
        s = re.sub(r"\s+", " ", re.sub(r"^total\s+", "", s))
        # "Earnings (Loss) Per Share Attributable to Intel—Diluted"
        return f"diluted {s[:-7].rstrip(' —-,')}" if s.endswith("diluted") else s

    if NON_GAAP.search(label) or NON_GAAP.search(parent):
        return None
    for cand in (norm(label), norm(f"{label} {parent}"), norm(f"{parent} {label}")):
        for syn in (cand, f"total {cand}"):
            if syn in SYNONYMS:
                return SYNONYMS[syn], "loss" in syn
    return None


def _table_claims(line: str, base: int, t: _Table) -> list[ClaimDraft]:
    cells = line.split(" | ")
    offsets, pos = [], 0
    for c in cells:
        offsets.append((pos, pos + len(c)))
        pos += len(c) + 3
    parsed = [parse_cell(c) for c in cells[1:]]
    if t.skip or not parsed or parsed[0] is None:
        return []
    label = _row_label(cells[0], t.parent)
    if label is None:
        return []
    metric, loss = label
    raw, suffix = parsed[0]
    unit = CONCEPTS[metric].unit
    if unit == "percent" and (suffix == "%" or (abs(raw) <= 100 and "." in cells[1])):
        value = raw
    elif unit == "USD/share":
        if "." not in cells[1]:
            return []  # per-share amounts print cents; "(11)" here is a percent change
        value = raw
    else:
        if t.scale is None:
            return []
        if unit == "percent":
            if metric != "gross_margin":
                return []
            metric, unit = "gross_profit", "USD"
        value = scaled(raw, t.scale)
    if loss:
        value = -abs(value)
    period = t.period
    out = [ClaimDraft(text=line[:offsets[1][1]], span_start=base, span_end=base + offsets[1][1],
                      claim_type="reported_result", metric=metric, value=value, unit=unit,
                      target_fiscal_year=period[0], target_fiscal_period=period[1])]
    if t.header and len(t.header) == len(cells):
        for i, h in enumerate(t.header):
            comp = {"Q/Q": "qoq", "Y/Y": "yoy"}.get(h.strip().upper())
            p = parsed[i - 1] if i else None
            if comp and p and p[1] in ("%", "pts"):
                out.append(ClaimDraft(
                    text=line[:offsets[i][1]], span_start=base, span_end=base + offsets[i][1],
                    claim_type="comparison", metric=metric, value=p[0], unit="percent",
                    target_fiscal_year=period[0], target_fiscal_period=period[1], comparison=comp))
    return out


# ---------------------------------------------------------------- entry points


def extract_claims(text: str, *, fiscal_year: int, fiscal_period: str,
                   speaker_roles: list[tuple[int, int, str]] | None = None,
                   fiscal_year_end: str | None = None, use_ner: bool = True,
                   on_progress: Callable[[llm.Usage], None] | None = None) -> list[ClaimDraft]:
    """All claims in `text`, which is a release (load_text) or a concatenated transcript.

    fiscal_year/fiscal_period: the quarter the document reports. fiscal_year_end (MMDD,
    company.fiscal_year_end) resolves "December quarter" and "quarter ended <date>".
    speaker_roles: (char_start, char_end, role); analyst and operator turns are skipped.
    Claims whose metric is outside CONCEPTS are returned as "qualitative" (see module doc).
    on_progress: called with the LLM pass's usage per batch of answers (refine.refine).

    Dedupe and role filtering run before the LLM pass, so no call is spent on a restatement or
    an analyst turn, and again after it, for claims the pass made identical.
    """
    from ledgerline.extract.refine import refine

    cands = _finish(_demote_partial_revenue(extract_candidates(
        text, fiscal_year=fiscal_year, fiscal_period=fiscal_period, fiscal_year_end=fiscal_year_end,
        use_ner=use_ner)), speaker_roles)
    cands, _ = refine(text, cands, fiscal_year, fiscal_period, fiscal_year_end, on_progress)
    return _finish([d if d.metric or d.claim_type == "qualitative"
                    else d.model_copy(update={"claim_type": "qualitative", "comparison": None})
                    for d in cands], None)


def extract_candidates(text: str, *, fiscal_year: int, fiscal_period: str,
                       fiscal_year_end: str | None = None, use_ner: bool = True) -> list[ClaimDraft]:
    """Every claim with the type its wording implies, metric possibly None, not deduplicated.
    extract_claims() is the contract; this is what the typing eval measures."""
    fy, fp, fye = fiscal_year, fiscal_period, fiscal_year_end
    drafts: list[ClaimDraft] = []
    ctx = _Ctx()
    titles: list[str] = []  # short prose lines since the last table ended
    header_rows: list[str] = []
    table: _Table | None = None
    parent = ""  # "Earnings per share:" above "Diluted | 2.02"
    pos = 0
    for line in text.split("\n"):
        base, pos = pos, pos + len(line) + 1
        if " | " in line:
            cells = line.split(" | ")
            # header row: no numbers, or only column years ("2024 | 2025")
            if not any(_has_value(c) for c in cells[1:]):
                if table is not None and table.rows_seen:
                    titles, header_rows = [], []  # a new table right after the last one
                header_rows.append(line)
                table = _table_header(titles, header_rows, fy, fp)
                table.header = cells
                continue
            if table is None:
                table = _table_header(titles, header_rows, fy, fp)
            if not table.rows_seen and ctx.guidance and not ctx.skip:
                table.guidance = _guidance_period(titles, header_rows, ctx, fy, fp, fye)
            table.rows_seen, table.parent = True, parent
            drafts += _guidance_row(line, base, table) if table.guidance else _table_claims(line, base, table)
            continue
        if re.match(r"\(\d{1,2}\)|\([a-z]\)", line):
            continue  # footnote: qualifies a table cell, not a management claim
        label_row = _label_row(line)
        if label_row:
            parent = line
        if table is not None and table.rows_seen:
            if label_row:
                continue
            table, titles, header_rows, parent = None, [], [], ""  # prose: the table is over
        if len(line) < 120:
            titles = (titles + [line])[-3:]
        if (h := _heading_ctx(line, fy, fp, fye)) is not None:
            ctx = h
            continue
        if ctx.skip or _STRONG_BOILER.search(line):
            continue
        for s, e in _sentences(line):
            drafts += _sentence_claims(line[s:e], base + s, ctx, fy, fp, fye, use_ner)
    return drafts


def _demote_partial_revenue(drafts: list[ClaimDraft]) -> list[ClaimDraft]:
    """A company has one revenue per period. A dollar revenue figure under half the largest one
    stated for the same target period is a segment's, product's or customer group's ("revenue
    to $4 billion" for AI infrastructure next to $63 billion company guidance): metric None.
    ponytail: trusts the largest figure; if that one is itself mis-scoped, the true total is
    demoted instead. A per-document total from the income statement table would be stricter."""
    def usd_revenue(d: ClaimDraft) -> bool:
        return d.metric == "revenue" and d.unit == "USD" and d.value is not None and d.value > 0

    top: dict[tuple, float] = {}
    for d in filter(usd_revenue, drafts):
        k = (d.target_fiscal_year, d.target_fiscal_period)
        top[k] = max(top.get(k, 0.0), d.value_high or d.value)
    return [d.model_copy(update={"metric": None}) if usd_revenue(d)
            and (d.value_high or d.value) < 0.5 * top[(d.target_fiscal_year, d.target_fiscal_period)] else d
            for d in drafts]


def _finish(drafts: list[ClaimDraft], roles: list[tuple[int, int, str]] | None) -> list[ClaimDraft]:
    out, spans, facts = [], set(), set()
    unmapped: dict[tuple, list[tuple[int, int]]] = {}  # metric-None fact -> spans kept
    for d in drafts:
        fact = (d.claim_type, d.metric, d.value, d.value_high, d.unit, d.comparison,
                d.target_fiscal_year, d.target_fiscal_period)
        if (d.span_start, d.span_end) in spans or (d.metric and fact in facts):
            continue  # the same number restated (headline bullet, then body, then table)
        # Without a metric the same value elsewhere may be another quantity, so only an
        # overlapping span is the same claim: "sales increased 2% (2% ..." read twice.
        if d.metric is None and any(a < d.span_end and d.span_start < b for a, b in unmapped.get(fact, ())):
            continue
        if roles is not None:
            role = next((r for a, b, r in roles if a <= d.span_start < b), None)
            if role in ("analyst", "operator"):
                continue
            d = d.model_copy(update={"speaker_role": role})
        spans.add((d.span_start, d.span_end))
        facts.add(fact)
        if d.metric is None:
            unmapped.setdefault(fact, []).append((d.span_start, d.span_end))
        out.append(d)
    return out


def _write(s: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    res = s.execute(insert(Claim).values(rows).on_conflict_do_nothing(constraint="uq_claim_span")
                    .returning(Claim.id))
    return len(res.all())


def _row(d: ClaimDraft, version: str, **src) -> dict:
    return d.model_dump() | src | {"extractor_version": version}


_RESULTS = re.compile(r"\b(?:report(?:s|ed|ing)?|announce[sd]?)\b[^.\n]{0,80}\bresults\b", re.IGNORECASE)


def is_earnings_release(text: str) -> bool:
    """An exhibit that reports a quarter's results: a results announcement in its opening and at
    least ten dollar figures. An event release filed as the quarter's EX-99.1 (NKE's CFO
    transition, filing 176) has neither, and any claim taken from it would be noise."""
    money = sum(v.kind in ("money", "per_share") for v in parse_values(text))
    return money >= 10 and bool(_RESULTS.search(text[:3000]))


def extract_for_filing(s: Session, filing_id: int, use_ner: bool = True,
                       on_progress: Callable[[llm.Usage], None] | None = None) -> int:
    """Claims from an 8-K EX-99.1 release; returns rows inserted (0 on a re-run).
    on_progress: see extract_claims (a heartbeat and cost hook for long LLM passes)."""
    from ledgerline.ingest.documents import load_text

    f = s.get(Filing, filing_id)
    if f is None or f.fiscal_year is None or f.fiscal_period is None:
        raise ValueError(f"filing {filing_id} missing or has no fiscal period")
    company = s.get(Company, f.company_id)
    text = load_text(f)
    if not is_earnings_release(text):
        log.warning("filing %s is not an earnings release (no results content): nothing extracted", filing_id)
        return 0
    version = extractor_version()
    drafts = extract_claims(text, fiscal_year=f.fiscal_year, fiscal_period=f.fiscal_period,
                            fiscal_year_end=company.fiscal_year_end, use_ner=use_ner,
                            on_progress=on_progress)
    return _write(s, [_row(d, version, speaker_role="executive", company_id=f.company_id,
                           fiscal_year=f.fiscal_year, fiscal_period=f.fiscal_period,
                           filing_id=f.id, segment_id=None) for d in drafts])


def extract_for_call(s: Session, call_audio_id: int, use_ner: bool = True,
                     on_progress: Callable[[llm.Usage], None] | None = None) -> int:
    """Claims from a transcribed call. Spans are stored relative to the segment's own text,
    which is what audio.pipeline.locate() takes to recover the timestamps.
    on_progress: see extract_claims."""
    from ledgerline.audio.pipeline import segment_text

    call = s.get(CallAudio, call_audio_id)
    if call is None or call.company_id is None or call.fiscal_year is None or call.fiscal_period is None:
        raise ValueError(f"call {call_audio_id} missing, not a company call, or has no fiscal period")
    text, seg_map = segment_text(s, call_audio_id)
    role_of = dict(s.execute(select(Segment.id, Segment.role)
                             .where(Segment.call_audio_id == call_audio_id)).all())
    company = s.get(Company, call.company_id)
    version = extractor_version()
    drafts = extract_claims(text, fiscal_year=call.fiscal_year, fiscal_period=call.fiscal_period,
                            speaker_roles=[(a, b, role_of[i]) for i, a, b in seg_map],
                            fiscal_year_end=company.fiscal_year_end, use_ner=use_ner,
                            on_progress=on_progress)
    rows = []
    for d in drafts:
        seg = next(((i, a) for i, a, b in seg_map if a <= d.span_start and d.span_end <= b), None)
        if seg is None:
            continue  # a span across two turns cannot be cited to one speaker
        rows.append(_row(d, version, span_start=d.span_start - seg[1], span_end=d.span_end - seg[1],
                         company_id=call.company_id, fiscal_year=call.fiscal_year,
                         fiscal_period=call.fiscal_period, filing_id=None, segment_id=seg[0]))
    return _write(s, rows)


def link_across_quarters(s: Session, company_id: int) -> int:
    """Point each actual at the guidance it can be scored against (FR-33/34).

    Match is exact on (metric, target year, target period, comparison-or-not): a reported
    level answers level guidance, a yoy growth claim answers growth guidance. The guidance
    must come from an earlier source quarter; the most recent such guidance wins (the last
    word management gave before the actual), then the lowest id. Returns claims linked.
    """
    version = extractor_version()
    cols = (Claim.id, Claim.metric, Claim.target_fiscal_year, Claim.target_fiscal_period,
            Claim.comparison, Claim.fiscal_year, Claim.fiscal_period)
    base = select(*cols).where(Claim.company_id == company_id, Claim.metric.is_not(None),
                               Claim.target_fiscal_year.is_not(None),
                               Claim.extractor_version == version)
    # Recompute from scratch so a re-run also clears links whose guidance no longer exists.
    s.execute(update(Claim).where(Claim.company_id == company_id,
                                  Claim.extractor_version == version,
                                  Claim.linked_claim_id.is_not(None)).values(linked_claim_id=None))
    guidance: dict[tuple, list] = {}
    for gid, metric, ty, tp, comp, fy, fp in s.execute(base.where(Claim.claim_type == "guidance")):
        guidance.setdefault((metric, ty, tp, comp), []).append((_key(fy, fp), gid))
    actuals = s.execute(base.where(Claim.claim_type.in_(("reported_result", "comparison")))).all()
    links = {}
    for aid, metric, ty, tp, comp, fy, fp in actuals:
        earlier = [g for g in guidance.get((metric, ty, tp, comp), []) if g[0] < _key(fy, fp)]
        if earlier:
            links[aid] = max(earlier, key=lambda g: (g[0], -g[1]))[1]
    for aid, gid in links.items():
        s.execute(update(Claim).where(Claim.id == aid).values(linked_claim_id=gid))
    return len(links)

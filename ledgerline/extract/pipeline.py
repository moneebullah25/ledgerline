"""S4 claim extraction: text -> typed, normalized, span-anchored ClaimDraft rows.

Offline and rules-based (FR-31): an optional Haiku refinement pass is not implemented, so
what is measured here is exactly what runs. Per line of the source text:

  table row ("Revenue | $96,221 | ...")  -> reported/comparison claims, only when the table
                                            header states both the scale and the period
  safe-harbor / boilerplate              -> dropped before any parsing
  heading or lead-in ("Outlook", "...as follows:") -> section context (guidance?, period)
  prose sentence                         -> one claim per value, typed by local cues;
                                            no value but evaluative language -> qualitative

Unmapped metrics: a value whose metric is outside CONCEPTS, or narrowed to a segment
("Data Center revenue"), is kept as claim_type "qualitative" with metric None and its value,
so the filing-RAG agent can still check it. Non-GAAP and constant-currency values are
dropped: no SEC filing states them as reported facts, so they can only ever be unverifiable.
"""

import re
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ledgerline.concepts import CONCEPTS
from ledgerline.extract.normalize import (
    NON_GAAP,
    SYNONYMS,
    Mention,
    Value,
    _qualified,
    find_periods,
    hedging_score,
    match_metrics,
    parse_cell,
    parse_values,
    resolve_period,
    scaled,
)
from ledgerline.models import CallAudio, Claim, Company, Filing, Segment
from ledgerline.schemas import ClaimDraft

EXTRACTOR_VERSION = "s4-rules-1"

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
_PAST = re.compile(r"\b(?:was|were|reported|grew|increased|decreased|declined|rose|fell|delivered"
                   r"|achieved|generated|posted|returned|had|exceeded)\b", re.IGNORECASE)
_CHANGE_BEFORE = re.compile(
    r"\b(?P<cue>up|down|increased?|decreased?|increase of|decrease of|grew|grow|growth(?: of)?"
    r"|rose|fell|declined?|decline of|expanded|contracted|improved|higher|lower)"
    r"(?:\s+(?:by|of|approximately|about|nearly|more than|over|roughly|to))*\s*$",
    re.IGNORECASE,
)
_CHANGE_AFTER = re.compile(r"^\s*(?:year[\s-]over[\s-]year\s+|yoy\s+|sequential\s+)?"
                           r"(?:growth|increase|decrease|decline|improvement)\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"down|decrease|decline|fell|lower|contracted", re.IGNORECASE)
_QOQ = re.compile(r"previous quarter|prior quarter|sequential|quarter[\s-]over[\s-]quarter|\bq/q\b"
                  r"|last quarter", re.IGNORECASE)
_YOY = re.compile(r"year[\s-]over[\s-]year|a year ago|\byoy\b|\by/y\b|prior year|last year|same period",
                  re.IGNORECASE)
_BASELINE = re.compile(r"\b(?:compared (?:with|to)|versus|vs\.?)\s", re.IGNORECASE)
_BASELINE_END = re.compile(r";|,\s+and\b")
_RELATIVE = re.compile(r"(?:last|prior|previous|same period of|a year ago|corresponding period of)\s*$",
                       re.IGNORECASE)
_DROP_AFTER = re.compile(r"^[^,;]{0,25}?(?:constant currency|currency[\s-]neutral)"
                         r"|^\s*(?:benefit|charge|impact|headwind|tailwind)", re.IGNORECASE)
# A component of the number, not the number: "including a favorable impact of $0.11".
_COMPONENT = re.compile(r"(?:\b(?:includ(?:ing|ed)|impact|benefit|charge|effect)\b[^,;$%]*|\bfrom\s*"
                        r"|\bprior (?:range|outlook|guidance|expectations?)\b[^,;$%]*)$", re.IGNORECASE)
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
_SKIP_HEADING = re.compile(r"non-gaap|definition|forward-looking|reconciliation|safe harbor"
                           r"|about [A-Z]|cautionary", re.IGNORECASE)
_BULLET = "•◦▪·-–*"
_ABBREV = {"inc", "corp", "co", "ltd", "u.s", "no", "vs", "st", "jr", "mr", "ms", "dr", "p.m",
           "a.m", "calif", "wash", "ore", "approx", "e.g", "i.e", "n.a", "u.k", "dept"}

# ---------------------------------------------------------------- helpers


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
    return _Ctx(guidance, period, segment=not _NEUTRAL_HEADING.search(s),
                skip=bool(_SKIP_HEADING.search(s)))


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
            if m.metric and _qualified(sent[gs:m.start]):
                m.metric = None  # GLiNER's wider span carries a segment qualifier
        if not inside and not any(m.start < ge and gs < m.end for m in ms):
            ms.append(Mention(gs, ge, None, bool(NON_GAAP.search(sent[max(0, gs - 20):ge]))))
    return sorted(ms, key=lambda m: m.start)


def _baseline_zones(sent: str) -> list[tuple[int, int]]:
    zones = []
    for m in _BASELINE.finditer(sent):
        end = _BASELINE_END.search(sent, m.end())
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


def _sentence_claims(sent: str, base: int, ctx: _Ctx, fy: int, fp: str, fye: str | None,
                     use_ner: bool) -> list[ClaimDraft]:
    if _BOILER.search(sent):
        return []
    values = parse_values(sent)
    mentions = _mentions(sent, use_ner) if (values or _QUAL_CUE.search(sent)) else []
    if ctx.segment:
        for m in mentions:
            m.metric = None
    zones = _baseline_zones(sent)
    drafts: list[ClaimDraft] = []
    prev_end = 0
    last_end: dict[int, int] = {}
    for i, v in enumerate(values):
        nxt = values[i + 1].start if i + 1 < len(values) else len(sent)
        post_end = min(nxt, (re.search(r"[,;.](?:\s|$)", sent[v.end:]) or re.search("$", sent[v.end:])).start()
                       + v.end)
        before, after = sent[prev_end:v.start], sent[v.end:nxt]
        prev_end = v.end
        if any(a <= v.start < b for a, b in zones) or _COMPONENT.search(before):
            continue  # prior-period baseline or an adjustment, not a claim about this period
        if _DROP_AFTER.search(after) or re.search(r"\bwould have been\s*$", before):
            continue  # constant currency, a component, or a counterfactual
        change = v.kind in ("percent", "pp", "flat") and bool(
            v.kind == "flat" or _CHANGE_BEFORE.search(before[-40:]) or _CHANGE_AFTER.search(after))
        if v.kind in ("money", "per_share") and _CHANGE_BEFORE.search(before[-25:]):
            continue  # dollar deltas ("an increase of $2.6 billion") are not stored
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
        if re.search(r"would have|constant currency|currency[\s-]neutral", between, re.IGNORECASE):
            continue
        last_end[id(mention)] = v.end
        ng_window = sent[mention.end if mention.end <= v.start else v.end:v.start] + sent[v.end:post_end]
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
        if target is None and ctx.period and ctx.guidance == forward:
            target = ctx.period
        if target is None and not forward:
            target = (fy, fp)
        if target and not change and _key(*target) > _key(fy, fp):
            forward = True
        claim_type = "guidance" if forward else "comparison" if change else "reported_result"
        comparison = None
        if change:
            comparison = "qoq" if _QOQ.search(after) and not _YOY.search(after[:_QOQ.search(after).start()]) else "yoy"
        value, value_high = v.value, v.value_high
        cue_before = _CHANGE_BEFORE.search(before[-40:]) if change else None
        if cue_before and _NEGATIVE.match(cue_before["cue"]):
            value, value_high = (-value_high, -value) if value_high is not None else (-value, None)

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
    first = re.search(r"three months ended|twelve months ended|(?:fiscal )?year ended|six months ended"
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
    elif first[0].lower().startswith("three"):
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


def _row_label(label: str, parent: str) -> str | None:
    def norm(s: str) -> str:
        s = re.sub(r"\([^)]*\)|\*|\d+$", "", s).strip().rstrip(":").strip().lower()
        return re.sub(r"^total\s+", "", s)

    if NON_GAAP.search(label) or NON_GAAP.search(parent):
        return None
    for cand in (norm(label), norm(f"{label} {parent}"), norm(f"{parent} {label}")):
        if cand in SYNONYMS:
            return SYNONYMS[cand]
        if f"total {cand}" in SYNONYMS:
            return SYNONYMS[f"total {cand}"]
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
    metric = _row_label(cells[0], t.parent)
    if metric is None:
        return []
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
                   fiscal_year_end: str | None = None, use_ner: bool = True) -> list[ClaimDraft]:
    """All claims in `text`, which is a release (load_text) or a concatenated transcript.

    fiscal_year/fiscal_period: the quarter the document reports. fiscal_year_end (MMDD,
    company.fiscal_year_end) resolves "December quarter" and "quarter ended <date>".
    speaker_roles: (char_start, char_end, role); analyst and operator turns are skipped.
    Claims whose metric is outside CONCEPTS are returned as "qualitative" (see module doc).
    """
    cands = extract_candidates(text, fiscal_year=fiscal_year, fiscal_period=fiscal_period,
                               fiscal_year_end=fiscal_year_end, use_ner=use_ner)
    return _finish([d if d.metric or d.claim_type == "qualitative"
                    else d.model_copy(update={"claim_type": "qualitative", "comparison": None})
                    for d in cands], speaker_roles)


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
            if all(parse_cell(c) is None or re.fullmatch(r"(?:19|20)\d\d", c) for c in cells[1:]):
                if table is not None and table.rows_seen:
                    titles, header_rows = [], []  # a new table right after the last one
                header_rows.append(line)
                table = _table_header(titles, header_rows, fy, fp)
                table.header = cells
            else:
                if table is None:
                    table = _table_header(titles, header_rows, fy, fp)
                table.rows_seen, table.parent = True, parent
                drafts += _table_claims(line, base, table)
            continue
        if re.match(r"\(\d{1,2}\)|\([a-z]\)", line):
            continue  # footnote: qualifies a table cell, not a management claim
        label_row = line.endswith(":") and len(line) < 60
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


def _finish(drafts: list[ClaimDraft], roles: list[tuple[int, int, str]] | None) -> list[ClaimDraft]:
    out, spans, facts = [], set(), set()
    for d in drafts:
        fact = (d.claim_type, d.metric, d.value, d.value_high, d.comparison,
                d.target_fiscal_year, d.target_fiscal_period)
        if (d.span_start, d.span_end) in spans or (d.metric and fact in facts):
            continue  # the same number restated (headline bullet, then body, then table)
        if roles is not None:
            role = next((r for a, b, r in roles if a <= d.span_start < b), None)
            if role in ("analyst", "operator"):
                continue
            d = d.model_copy(update={"speaker_role": role})
        spans.add((d.span_start, d.span_end))
        facts.add(fact)
        out.append(d)
    return out


def _write(s: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    res = s.execute(insert(Claim).values(rows).on_conflict_do_nothing(constraint="uq_claim_span")
                    .returning(Claim.id))
    return len(res.all())


def _row(d: ClaimDraft, **src) -> dict:
    return d.model_dump() | src | {"extractor_version": EXTRACTOR_VERSION}


def extract_for_filing(s: Session, filing_id: int, use_ner: bool = True) -> int:
    """Claims from an 8-K EX-99.1 release; returns rows inserted (0 on a re-run)."""
    from ledgerline.ingest.documents import load_text

    f = s.get(Filing, filing_id)
    if f is None or f.fiscal_year is None or f.fiscal_period is None:
        raise ValueError(f"filing {filing_id} missing or has no fiscal period")
    company = s.get(Company, f.company_id)
    drafts = extract_claims(load_text(f), fiscal_year=f.fiscal_year, fiscal_period=f.fiscal_period,
                            fiscal_year_end=company.fiscal_year_end, use_ner=use_ner)
    return _write(s, [_row(d, speaker_role="executive", company_id=f.company_id,
                           fiscal_year=f.fiscal_year, fiscal_period=f.fiscal_period,
                           filing_id=f.id, segment_id=None) for d in drafts])


def extract_for_call(s: Session, call_audio_id: int, use_ner: bool = True) -> int:
    """Claims from a transcribed call. Spans are stored relative to the segment's own text,
    which is what audio.pipeline.locate() takes to recover the timestamps."""
    from ledgerline.audio.pipeline import segment_text

    call = s.get(CallAudio, call_audio_id)
    if call is None or call.company_id is None or call.fiscal_year is None or call.fiscal_period is None:
        raise ValueError(f"call {call_audio_id} missing, not a company call, or has no fiscal period")
    text, seg_map = segment_text(s, call_audio_id)
    role_of = dict(s.execute(select(Segment.id, Segment.role)
                             .where(Segment.call_audio_id == call_audio_id)).all())
    company = s.get(Company, call.company_id)
    drafts = extract_claims(text, fiscal_year=call.fiscal_year, fiscal_period=call.fiscal_period,
                            speaker_roles=[(a, b, role_of[i]) for i, a, b in seg_map],
                            fiscal_year_end=company.fiscal_year_end, use_ner=use_ner)
    rows = []
    for d in drafts:
        seg = next(((i, a) for i, a, b in seg_map if a <= d.span_start and d.span_end <= b), None)
        if seg is None:
            continue  # a span across two turns cannot be cited to one speaker
        rows.append(_row(d, span_start=d.span_start - seg[1], span_end=d.span_end - seg[1],
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
    cols = (Claim.id, Claim.metric, Claim.target_fiscal_year, Claim.target_fiscal_period,
            Claim.comparison, Claim.fiscal_year, Claim.fiscal_period)
    base = select(*cols).where(Claim.company_id == company_id, Claim.metric.is_not(None),
                               Claim.target_fiscal_year.is_not(None),
                               Claim.extractor_version == EXTRACTOR_VERSION)
    # Recompute from scratch so a re-run also clears links whose guidance no longer exists.
    s.execute(update(Claim).where(Claim.company_id == company_id,
                                  Claim.extractor_version == EXTRACTOR_VERSION,
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

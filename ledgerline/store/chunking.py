"""FR-20 section-aware chunking. Pure: text in, chunk drafts out.

Offsets index into the exact string passed in (ingest.documents.load_text), so a chunk is
always text[char_start:char_end] and a citation can be highlighted in the source.
"""

import re
from dataclasses import dataclass

MAX_TOKENS = 800
OVERLAP_TOKENS = 100

# Token count is a word/punctuation approximation of BERT pre-tokenization (what bge's
# WordPiece tokenizer splits on before sub-word splitting). It undercounts WordPiece
# tokens (rare words split further), so it is a lower bound. Chosen to keep chunking pure
# and model-free; the measured ratio against the real tokenizer is in the eval details.
_TOKEN = re.compile(r"\w+|[^\w\s]")

_PART = re.compile(r"^[ \t]*PART[ \t]+(IV|III|II|I)\b[ \t]*(?:[—–:.\-][^\n]*)?$", re.I | re.M)
# "Item 2. Title", "ITEM 1A — Title", or a bare "Item 2." with the title on the next line.
_ITEM = re.compile(r"^[ \t]*ITEM[ \t]+(\d{1,2}[A-C]?)\b[ \t]*[.:—–\-]?[ \t]*([^\n]*)$", re.I | re.M)
_MAX_HEADER_LINE = 200

# 8-K press releases have no Item structure: a heading is a short, unpunctuated line directly
# followed by a paragraph, or a financial-statement title.
_STATEMENT_TITLE = re.compile(
    r"^(condensed |unaudited )*(consolidated )?(statements? of|balance sheets?|reconciliation)",
    re.I,
)


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    section_path: str
    char_start: int
    char_end: int
    text: str
    token_count: int


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _next_line(text: str, pos: int) -> str:
    """First non-blank line starting at pos (the title of a bare "Item 2." header)."""
    for line in text[pos:pos + 400].split("\n"):
        if line.strip():
            return line.strip()
    return ""


def _periodic_sections(text: str) -> list[tuple[int, str]]:
    """(start offset, section_path) for 10-K/10-Q Part and Item headers.

    The table of contents repeats every header, so for each (part, item) only the LAST
    occurrence is kept: the TOC copy then falls into the preceding section instead of
    producing a run of empty or bogus sections.
    """
    cands: list[tuple[int, str, str]] = []  # (offset, kind, value)
    for m in _PART.finditer(text):
        cands.append((m.start(), "part", m.group(1).upper()))
    for m in _ITEM.finditer(text):
        if len(m.group(0)) > _MAX_HEADER_LINE:
            continue
        title = m.group(2).strip() or _next_line(text, m.end())
        # Real headers are title-cased; "Item 1A of the 2025 Form 10-K" is a cross-reference.
        if not title[:1].isupper() or len(title) > _MAX_HEADER_LINE or _ITEM.match(title):
            continue
        title = re.sub(r"\s+", " ", title).rstrip(" .0123456789")  # drop TOC page numbers
        cands.append((m.start(), "item", f"Item {m.group(1).upper()}. {title[:100]}"))
    cands.sort()

    last: dict[tuple[str, str], tuple[int, str, str]] = {}
    part = ""
    for off, kind, value in cands:
        if kind == "part":
            part = value
            last[("part", value)] = (off, kind, value)
        else:
            number = value.split(".")[0]
            last[(part, number)] = (off, kind, value)

    out: list[tuple[int, str]] = []
    part = ""
    for off, kind, value in sorted(last.values()):
        if kind == "part":
            part = f"Part {value}"
            out.append((off, part))
        else:
            out.append((off, f"{part} > {value}" if part else value))
    return out


def _press_release_sections(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    lines = [(m.start(), m.group(0).strip()) for m in re.finditer(r"[^\n]+", text)]
    lines = [(o, s) for o, s in lines if s]
    for (off, line), (_, nxt) in zip(lines, lines[1:] + [(0, "")]):
        short = 2 <= len(line.split()) and len(line) <= 100 and line[0].isupper()
        if short and (_STATEMENT_TITLE.match(line) or (line[-1] not in ".,;:" and len(nxt) >= 200)):
            out.append((off, line))
    return out


def _units(text: str, start: int, end: int, max_tokens: int) -> list[tuple[int, int, int]]:
    """Split text[start:end] into (start, end, tokens) units no larger than max_tokens:
    lines first, then sentences, then fixed word windows for pathological run-ons."""
    out = []
    for m in re.finditer(r"[^\n]+", text[start:end]):
        for s, e in _split(text, start + m.start(), start + m.end(), max_tokens):
            stripped = text[s:e]
            lead = len(stripped) - len(stripped.lstrip())
            s, e = s + lead, s + len(stripped.rstrip())
            if e > s:
                out.append((s, e, count_tokens(text[s:e])))
    return out


def _split(text: str, s: int, e: int, max_tokens: int) -> list[tuple[int, int]]:
    if count_tokens(text[s:e]) <= max_tokens:
        return [(s, e)]
    pieces, cur = [], s
    for m in re.finditer(r"(?<=[.!?])\s+", text[s:e]):
        pieces.append((cur, s + m.start()))
        cur = s + m.end()
    pieces.append((cur, e))
    out = []
    for ps, pe in pieces:
        if count_tokens(text[ps:pe]) <= max_tokens:
            out.append((ps, pe))
            continue
        toks = list(_TOKEN.finditer(text, ps, pe))
        for i in range(0, len(toks), max_tokens):
            window = toks[i:i + max_tokens]
            out.append((window[0].start(), window[-1].end()))
    return out


def chunk_document(
    text: str, form_type: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS
) -> list[ChunkDraft]:
    """Greedy-pack line/sentence units into ~max_tokens chunks with ~overlap tokens carried
    over, restarting at every section boundary so no chunk mixes two sections."""
    if form_type.startswith("8-K"):
        sections = _press_release_sections(text)
        preamble = "Press release"
    else:
        sections = _periodic_sections(text)
        preamble = "Cover"
    bounds = sections if sections and sections[0][0] == 0 else [(0, preamble)] + sections
    chunks: list[ChunkDraft] = []
    for (start, path), (end, _) in zip(bounds, bounds[1:] + [(len(text), "")]):
        units = _units(text, start, end, max_tokens)
        if (start, path) in sections and "\n" not in text[start:end].strip():
            continue  # a bare header ("PART I — FINANCIAL INFORMATION") with no content
        i = 0
        while i < len(units):
            j, total = i, 0
            while j < len(units) and total + units[j][2] <= max_tokens:
                total += units[j][2]
                j += 1
            cs, ce = units[i][0], units[j - 1][1]
            chunks.append(ChunkDraft(
                len(chunks), path, cs, ce, text[cs:ce], count_tokens(text[cs:ce])
            ))
            if j == len(units):
                break
            # Step back over trailing units worth <= overlap tokens, always moving forward.
            k, carried = j, 0
            while k - 1 > i and carried + units[k - 1][2] <= overlap:
                k -= 1
                carried += units[k][2]
            i = k
    return chunks

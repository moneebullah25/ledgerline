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

# The optional tail allows "PART I — FINANCIAL INFORMATION", "PART I FINANCIAL INFORMATION"
# (capitalised title, case-sensitive) and TOC rows "Part I | 3"; not "Part I of this report".
_PART = re.compile(
    r"^[ \t]*PART[ \t]+(IV|III|II|I)\b(?:[ \t]*[—–:.|\-][^\n]{0,100}"
    r"|[ \t]+(?-i:[A-Z])[^\n]{0,100})?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# "Item 2. Title", "ITEM 1A — Title", or a bare "Item 2." with the title on the next line.
_ITEM = re.compile(
    r"^[ \t]*ITEM[ \t]+(\d{1,2}[A-C]?)\b[ \t]*[.:—–\-]?([^\n]*)$", re.IGNORECASE | re.MULTILINE
)
_MAX_HEADER_LINE = 200

# Documents without Item structure (8-K press releases, some 10-Qs): a heading is a short,
# unpunctuated line directly followed by a paragraph, or a financial-statement title.
_STATEMENT_TITLE = re.compile(
    r"^(condensed |unaudited )*(consolidated )?(statements? of|balance sheets?|reconciliation)",
    re.IGNORECASE,
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
    for line in text[pos : pos + 400].split("\n"):
        if line.strip():
            return line.strip()
    return ""


def _periodic_sections(text: str) -> list[tuple[int, str]]:
    """(start offset, section_path) for 10-K/10-Q Part and Item headers.

    Every header can occur several times: in the table of contents, as a running page
    header (MSFT), or as a cross-reference that happens to start a line. The real headers
    are the copies that appear in the filing's own Item order AND carry the document's
    text: of all chains of Item headers whose numbering ascends with position, the one
    covering the most characters wins. TOC entries cover a page number each, so a TOC chain
    loses to the body chain. Dropped copies fall into the surrounding section instead of
    becoming empty or bogus sections.
    """
    cands: list[tuple[int, int, str, str]] = []  # (offset, header end, kind, value)
    for m in _PART.finditer(text):
        cands.append((m.start(), m.end(), "part", m.group(1).upper()))
    for m in _ITEM.finditer(text):
        if len(m.group(0)) > _MAX_HEADER_LINE:
            continue
        # Headers laid out as table rows render as "Item 2. | Title | 13" (load_text joins
        # cells with " | "); the first non-empty cell after the number is the title.
        title, end = next((c.strip() for c in m.group(2).split("|") if c.strip()), ""), m.end()
        if not title:
            # A bare "Item 1" without punctuation is a running page header, not a heading.
            if not m.group(0).rstrip().endswith((".", ":", "—", "–", "-")):
                continue
            title = _next_line(text, m.end())
            at = text.find(title, m.end())
            end = at + len(title) if at >= 0 else m.end()
        # Real headers are title-cased; "Item 1A of the 2025 Form 10-K" is a cross-reference.
        if not title[:1].isupper() or len(title) > _MAX_HEADER_LINE or _ITEM.match(title):
            continue
        title = re.sub(r"\s+", " ", title).rstrip(" .0123456789")  # drop TOC page numbers
        cands.append((m.start(), end, "item", f"Item {m.group(1).upper()}. {title[:100]}"))
    cands.sort()

    # (offset, numbering key, part, value, chars covered): _heaviest_ascending_chain reads
    # the key at [1] and the weight at [-1].
    items = []
    part_offsets: dict[str, list[int]] = {}
    has_parts = any(kind == "part" for _, _, kind, _ in cands)
    part = ""
    for i, (off, end, kind, value) in enumerate(cands):
        if kind == "part":
            part = value
            part_offsets.setdefault(value, []).append(off)
            continue
        if not part and has_parts:
            continue  # above the first Part header: a TOC copy (or cover text)
        nxt = cands[i + 1][0] if i + 1 < len(cands) else len(text)
        number = value.split(".")[0].removeprefix("Item ")
        # +1 so an empty section ("Item 4. Mine Safety Disclosures" then "Item 5.") still
        # extends the chain.
        items.append((off, _order(part, number), part, value, len(text[end:nxt].strip()) + 1))

    out: list[tuple[int, str]] = []
    first_item: dict[str, int] = {}
    for off, _, part, value, _ in _heaviest_ascending_chain(items):
        out.append((off, f"Part {part} > {value}" if part else value))
        first_item.setdefault(part, off)
    # A Part header is the nearest copy above its first item: earlier copies are the TOC,
    # later ones running page headers.
    for part, item_off in first_item.items():
        above = [o for o in part_offsets.get(part, []) if o <= item_off]
        if above:
            out.append((max(above), f"Part {part}"))
    return sorted(out)


_ROMAN = {"": 0, "I": 1, "II": 2, "III": 3, "IV": 4}


def _order(part: str, number: str) -> tuple[int, int, str]:
    """Sort key of an Item in a filing's own numbering: Part II Item 1 follows Part I Item 4."""
    return _ROMAN[part], int(number.rstrip("ABC")), number[-1] if number[-1].isalpha() else ""


def _heaviest_ascending_chain(items: list[tuple]) -> list[tuple]:
    """items sorted by offset as (offset, key, ..., weight): the subsequence with strictly
    ascending keys and the largest total weight. O(n^2); n is tens of headers."""
    best = [it[-1] for it in items]
    prev = [-1] * len(items)
    for i in range(len(items)):
        for j in range(i):
            if items[j][1] < items[i][1] and best[j] + items[i][-1] > best[i]:
                best[i], prev[i] = best[j] + items[i][-1], j
    i = max(range(len(items)), key=best.__getitem__, default=-1)
    chain = []
    while i >= 0:
        chain.append(items[i])
        i = prev[i]
    return chain[::-1]


def _heading_sections(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    lines = [(m.start(), m.group(0).strip()) for m in re.finditer(r"[^\n]+", text)]
    lines = [(o, s) for o, s in lines if s]
    for (off, line), (_, nxt) in zip(lines, lines[1:] + [(0, "")]):
        # " | " marks a table row (load_text joins cells with it), not a heading.
        short = (
            2 <= len(line.split())
            and len(line) <= 100
            and line[0].isupper()
            and "|" not in line
            and line.lower() != "table of contents"
        )
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
            window = toks[i : i + max_tokens]
            out.append((window[0].start(), window[-1].end()))
    return out


def chunk_document(
    text: str, form_type: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS
) -> list[ChunkDraft]:
    """Greedy-pack line/sentence units into ~max_tokens chunks with ~overlap tokens carried
    over, restarting at every section boundary so no chunk mixes two sections."""
    if form_type.startswith("8-K"):
        sections, preamble = _heading_sections(text), "Press release"
    else:
        sections, preamble = _periodic_sections(text), "Cover"
        # Some filers (INTC, MCD) use no Item headers in the body, only a cross-reference
        # index at the end; split those by their own headings instead.
        if not sections or sections[0][0] > len(text) // 2:
            sections = _heading_sections(text)
    bounds = sections if sections and sections[0][0] == 0 else [(0, preamble)] + sections
    chunks: list[ChunkDraft] = []
    for (start, path), (end, _) in zip(bounds, bounds[1:] + [(len(text), "")]):
        if (start, path) in sections and "\n" not in text[start:end].strip():
            continue  # a bare header ("PART I — FINANCIAL INFORMATION") with no content
        units = _units(text, start, end, max_tokens)
        i = 0
        while i < len(units):
            j, total = i, 0
            while j < len(units) and total + units[j][2] <= max_tokens:
                total += units[j][2]
                j += 1
            cs, ce = units[i][0], units[j - 1][1]
            chunks.append(
                ChunkDraft(len(chunks), path, cs, ce, text[cs:ce], count_tokens(text[cs:ce]))
            )
            if j == len(units):
                break
            # Step back over trailing units worth <= overlap tokens, always moving forward.
            k, carried = j, 0
            while k - 1 > i and carried + units[k - 1][2] <= overlap:
                k -= 1
                carried += units[k][2]
            i = k
    return chunks

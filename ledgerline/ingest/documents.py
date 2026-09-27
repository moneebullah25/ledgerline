"""Deterministic stored document -> plain text.

Claim spans and chunk offsets index into the returned string, so it depends only on the
stored bytes: no clocks, no network, no parser choice left to autodetection.
"""

import re
import warnings
from pathlib import PurePath

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

from ledgerline.ingest.objects import read_object
from ledgerline.models import Filing

_BLOCK = (
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "section",
    "article",
    "center",
    "blockquote",
    "pre",
    "hr",
)
_HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
# Text a reader never sees must not become evidence (or an injection channel to the LLM).
# Only invisible elements holding letters or digits are dropped: filers also hide stray
# punctuation this way (Adobe's FY2025 10-K has "12 %%" with one "%" at font-size:0), and
# keeping those leaves the text, and so every stored chunk offset, byte-identical.
_INVISIBLE = re.compile(
    r"visibility\s*:\s*hidden|font-size\s*:\s*0+(?:\.0+)?(?![\d.])", re.IGNORECASE
)
_WORDLIKE = re.compile(r"\w")
_WS = re.compile(r"[ \t\r\f\v\xa0\u2002\u2003\u2009\u200b]+")
# Inline-XBRL documents are XHTML; the HTML parser is used on purpose so every filing goes
# through one code path.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
_BREAK = "\x1e"  # marks a block boundary; source newlines inside a paragraph are spaces


def load_text(filing: Filing) -> str:
    return html_to_text(read_object(filing.storage_path), PurePath(filing.storage_path).suffix)


def html_to_text(data: bytes, suffix: str = ".htm") -> str:
    if suffix.lower() not in (".htm", ".html", ".xml", ""):
        return _normalize(data.decode("utf-8", errors="replace"))
    soup = BeautifulSoup(data, "lxml")
    for tag in soup.find_all(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN):
        tag.decompose()
    for tag in soup.find_all(style=_INVISIBLE):
        if _WORDLIKE.search(tag.get_text()):
            tag.decompose()
    # Rows become one line so a number stays next to its label; innermost tables first.
    for tr in reversed(soup.find_all("tr")):
        tr.replace_with(NavigableString(_BREAK + _row(tr) + _BREAK))
    for tag in soup.find_all(_BLOCK):
        tag.insert_before(_BREAK)
        tag.insert_after(_BREAK)
    return _normalize(soup.get_text().replace("\n", " ").replace(_BREAK, "\n"))


def _row(tr) -> str:
    cells: list[str] = []
    for td in tr.find_all(["td", "th"]):
        t = _WS.sub(" ", td.get_text(" ")).strip()
        if not t:
            continue  # SEC tables use empty spacer cells everywhere
        # "$", "(" and ")" / "%" sit in their own cells in EDGAR tables; glue them to the number.
        if cells and (cells[-1] in ("$", "(", "($", "$(") or t in (")", "%", ")%", "%)")):
            cells[-1] += t
        else:
            cells.append(t)
    return " | ".join(cells)


def _normalize(text: str) -> str:
    lines = (_WS.sub(" ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)

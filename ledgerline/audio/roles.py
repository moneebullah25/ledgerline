"""Speaker roles and the prepared-remarks -> Q&A split (FR-12, FR-13). Pure text logic.

Whisper gives no speakers, so roles come from what people say: regex cues over each turn
seed a role for that turn's speaker and/or the next speaker, then every speaker cluster
takes the majority of its seeds. Works for earnings calls and FOMC press conferences.
"""

import re
from collections import Counter

# (cue, role of the speaker saying it, role of the next speaker)
_I = re.IGNORECASE
_SEEDS: list[tuple[re.Pattern, str | None, str | None]] = [
    (
        re.compile(r"\b(first|next|last|final|following) question (comes|is coming|is) from", _I),
        "operator",
        "analyst",
    ),
    (
        re.compile(
            r"\bpress (star|\*) ?(one|1)\b|\bladies and gentlemen\b"
            r"|\bthis concludes (today's|our|the) (conference|call|program)",
            _I,
        ),
        "operator",
        None,
    ),
    # "turn the call over to X, our CFO" -> X is an executive; handing back to the operator
    # says nothing about the next speaker.
    (
        re.compile(r"\bturn (the call |it |things )?(back )?over to (?!(the )?operator)", _I),
        None,
        "executive",
    ),
    # Separate cues so a statement that has several counts as several votes.
    (re.compile(r"\blook forward to (your|taking your) questions", _I), "executive", None),
    (re.compile(r"\bmy colleagues and I\b", _I), "executive", None),
    (re.compile(r"\bwe (reported|delivered|generated|achieved)\b", _I), "executive", None),
    (
        re.compile(
            r"\bthanks? (you )?for taking (my|our|the) questions?"
            r"|\b(thank you|thanks),? (mr\.? chair(man)?|madam chair|chair \w+)\b",
            _I,
        ),
        "analyst",
        None,
    ),
    # FOMC reporters open with "Steve Liesman, CNBC." / "Howard Schneider with Reuters."
    # (case-sensitive on purpose).
    (
        re.compile(
            r"^\W*(?:(?:hi|thank you|thanks)[.,]?\s+)?[A-Z][a-z]+ [A-Z][\w'-]+"
            r"(?: [A-Z][\w'-]+)?(?:,| with| at| from) (?:with |from )?(?:the )?[A-Z]"
        ),
        "analyst",
        None,
    ),
    # FOMC moderator calls on a reporter by name: "Steve." / "Thank you. Nick."
    (
        re.compile(
            r"^\W*(?:(?:thank you|thanks|okay)[.,]?\s+)?"
            r"(?!(?:Okay|Yes|No|Thanks|Thank|Right|Sure|So|Well|Great|Good)\b)"
            r"[A-Z][a-z]+(?: [A-Z][a-z]+)?\.?\W*$"
        ),
        "operator",
        "analyst",
    ),
]

# Only the instruction that starts Q&A, not the operator's early "there will be a Q&A
# session after the presentation", hence the now/begin/open anchor.
_QA = re.compile(
    r"\b(now|begin|open|start)\b[^.?!]{0,40}\b(question[- ]and[- ]answer|questions|q ?(&|and) ?a)\b"
    r"|\blook forward to (your|taking your) questions",
    _I,
)


def seed_roles(texts: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """(own, prev) role votes per turn: own[i] come from what turn i itself says, prev[i]
    from the turn before it ("our next question comes from ...")."""
    own: list[list[str]] = [[] for _ in texts]
    prev: list[list[str]] = [[] for _ in texts]
    for i, text in enumerate(texts):
        for cue, this, nxt in _SEEDS:
            if cue.search(text):
                if this:
                    own[i].append(this)
                if nxt and i + 1 < len(texts):
                    prev[i + 1].append(nxt)
    return own, prev


def assign_roles(texts: list[str], speakers: list[str]) -> list[str]:
    """A turn's own cues decide its role. Otherwise the speaker cluster's majority vote
    decides, own cues counting double: a next-speaker cue is only as good as the turn
    boundaries, and one reporter merged into the chair's cluster must not relabel it."""
    own, prev = seed_roles(texts)
    by_speaker: dict[str, Counter] = {}
    for spk, o, p in zip(speakers, own, prev, strict=True):
        by_speaker.setdefault(spk, Counter()).update(o + o + p)
    out = []
    for spk, o, p in zip(speakers, own, prev, strict=True):
        c = Counter(o) or by_speaker[spk] or Counter(p)
        out.append(c.most_common(1)[0][0] if c else "unknown")
    return out


def qa_start(texts: list[str]) -> int | None:
    """Index of the first Q&A turn: the one after the question-instruction phrase."""
    return next((i + 1 for i, t in enumerate(texts) if _QA.search(t)), None)


def sections(texts: list[str]) -> list[str]:
    start = qa_start(texts)
    return [
        "qa" if start is not None and i >= start else "prepared_remarks" for i in range(len(texts))
    ]

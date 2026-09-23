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
    (re.compile(r"\b(first|next|last|final|following) question (comes|is coming|is) from", _I),
     "operator", "analyst"),
    (re.compile(r"\bpress (star|\*) ?(one|1)\b|\bladies and gentlemen\b"
                r"|\bthis concludes (today's|our|the) (conference|call|program)", _I),
     "operator", None),
    # "turn the call over to X, our CFO" -> X is an executive; handing back to the operator
    # says nothing about the next speaker.
    (re.compile(r"\bturn (the call |it |things )?(back )?over to (?!(the )?operator)", _I),
     None, "executive"),
    (re.compile(r"\blook forward to (your|taking your) questions|\bmy colleagues and I\b"
                r"|\bwe (reported|delivered|generated|achieved)\b", _I),
     "executive", None),
    (re.compile(r"\bthanks? (you )?for taking (my|our|the) questions?"
                r"|\b(thank you|thanks),? (mr\.? chair(man)?|madam chair|chair \w+)\b", _I),
     "analyst", None),
    # FOMC reporters open with "Steve Liesman, CNBC." / "Howard Schneider with Reuters."
    # (case-sensitive on purpose).
    (re.compile(r"^\W*(?:(?:hi|thank you|thanks)[.,]?\s+)?[A-Z][a-z]+ [A-Z][\w'-]+"
                r"(?: [A-Z][\w'-]+)?(?:,| with| at| from) (?:with |from )?(?:the )?[A-Z]"),
     "analyst", None),
    # FOMC moderator calls on a reporter by name: "Steve." / "Thank you. Nick."
    (re.compile(r"^\W*(?:(?:thank you|thanks|okay)[.,]?\s+)?"
                r"(?!(?:Okay|Yes|No|Thanks|Thank|Right|Sure|So|Well|Great|Good)\b)"
                r"[A-Z][a-z]+(?: [A-Z][a-z]+)?\.?\W*$"), "operator", "analyst"),
]

# Only the instruction that starts Q&A, not the operator's early "there will be a Q&A
# session after the presentation", hence the now/begin/open anchor.
_QA = re.compile(
    r"\b(now|begin|open|start)\b[^.?!]{0,40}\b(question[- ]and[- ]answer|questions|q ?(&|and) ?a)\b"
    r"|\blook forward to (your|taking your) questions", _I)


def seed_roles(texts: list[str]) -> list[list[str]]:
    """Role votes per turn: votes[i] are the cues that say who turn i's speaker is."""
    votes: list[list[str]] = [[] for _ in texts]
    for i, text in enumerate(texts):
        for cue, this, nxt in _SEEDS:
            if cue.search(text):
                if this:
                    votes[i].append(this)
                if nxt and i + 1 < len(texts):
                    votes[i + 1].append(nxt)
    return votes


def assign_roles(texts: list[str], speakers: list[str]) -> list[str]:
    """Majority vote of seeds per speaker cluster; an unseeded cluster falls back to the
    turn's own seed, else unknown."""
    votes = seed_roles(texts)
    by_speaker: dict[str, Counter] = {}
    for spk, v in zip(speakers, votes, strict=True):
        by_speaker.setdefault(spk, Counter()).update(v)
    out = []
    for spk, v in zip(speakers, votes, strict=True):
        c = by_speaker[spk] or Counter(v)
        out.append(c.most_common(1)[0][0] if c else "unknown")
    return out


def qa_start(texts: list[str]) -> int | None:
    """Index of the first Q&A turn: the one after the question-instruction phrase."""
    return next((i + 1 for i, t in enumerate(texts) if _QA.search(t)), None)


def sections(texts: list[str]) -> list[str]:
    start = qa_start(texts)
    return ["qa" if start is not None and i >= start else "prepared_remarks"
            for i in range(len(texts))]

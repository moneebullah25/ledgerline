"""S5/S6 verification graph. Entry points: build.verify_claim, router.route."""

import hashlib
from functools import cache
from pathlib import Path

_PROMPTS = Path(__file__).parent / "prompts"


@cache
def load_prompt(name: str) -> tuple[str, str]:
    """(text, version). version = file stem + first 8 hex of the file's sha256, so any edit
    to a prompt changes the version recorded on every verdict it produced."""
    raw = (_PROMPTS / f"{name}.md").read_bytes()
    return raw.decode(), f"{name}-{hashlib.sha256(raw).hexdigest()[:8]}"

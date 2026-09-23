"""The one place Ledgerline calls Claude.

Model tiering (§10.8): cheap/high-volume steps on Haiku, evidence agents and the critic on
Sonnet, the adjudicator on Opus. Untrusted text (filings, transcripts) is always sent as a
delimited data block in the user turn, never in the system prompt (§16 prompt injection).
The stable system prompt carries a cache_control breakpoint; check cache_read_tokens in the
trace to confirm it hits (§10.8: a zero read rate is a bug).

When no Anthropic credentials are configured, available() is False and every caller uses
its deterministic offline implementation instead. Nothing here fabricates a response.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from pydantic import BaseModel

from ledgerline.config import settings

TIERS = {"fast": "claude-haiku-4-5", "standard": "claude-sonnet-5", "judge": "claude-opus-5"}

# USD per million tokens: input, output, cache read, cache write (5-minute TTL = 1.25x input).
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
    "claude-sonnet-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
}


@dataclass(frozen=True)
class Usage:
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        if self.model is None:
            return 0.0
        i, o, r, w = PRICES[self.model]
        return (
            self.input_tokens * i + self.output_tokens * o
            + self.cache_read_tokens * r + self.cache_write_tokens * w
        ) / 1e6


OFFLINE = Usage()


def available() -> bool:
    return settings().llm_enabled and bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


@lru_cache
def _client():
    import anthropic

    return anthropic.Anthropic()


def structured[T: BaseModel](
    tier: str, system: str, data: dict[str, str], instruction: str, schema: type[T],
    max_tokens: int = 4000,
) -> tuple[T, Usage]:
    """One schema-constrained call. `data` maps a label to untrusted text; each is wrapped
    in its own <document> block so instructions inside it are read as content, not commands.
    Raises ValueError if the model refuses or the output does not validate."""
    model = TIERS[tier]
    docs = "\n".join(
        f'<document label="{label}">\n{text}\n</document>' for label, text in data.items()
    )
    resp = _client().messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"{docs}\n\n{instruction}"}],
        output_format=schema,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise ValueError(f"{model} returned no valid {schema.__name__} ({resp.stop_reason})")
    u = resp.usage
    return resp.parsed_output, Usage(
        model=model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        cache_write_tokens=u.cache_creation_input_tokens or 0,
    )

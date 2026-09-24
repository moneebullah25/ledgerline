"""The one place Ledgerline calls Claude.

Model tiering (§10.8): cheap/high-volume steps on Haiku, evidence agents and the critic on
Sonnet, the adjudicator on Opus. Untrusted text (filings, transcripts) is always sent as a
delimited data block in the user turn, never in the system prompt (§16 prompt injection).
The stable system prompt carries a cache_control breakpoint; check cache_read_tokens in the
trace to confirm it hits (§10.8: a zero read rate is a bug).

When no Anthropic credentials are configured, available() is False and every caller uses
its deterministic offline implementation instead. Nothing here fabricates a response.
"""

import json
import os
import re
import secrets
import subprocess
import tempfile
import time
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
    cfg = settings()
    if not cfg.llm_enabled:
        return False
    if cfg.llm_backend == "claude_cli":
        return os.access(cfg.claude_bin, os.X_OK)
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


_TAG = re.compile(r"<(/?)(document)", re.IGNORECASE)


def _neutralise(text: str) -> str:
    """Untrusted text cannot open or close a data block: any <document... / </document...
    inside it (which also covers the per-call tag) loses its angle bracket."""
    return _TAG.sub(r"&lt;\1\2", text)


@lru_cache
def _client():
    import anthropic

    return anthropic.Anthropic()


def structured[T: BaseModel](
    tier: str, system: str, data: dict[str, str], instruction: str, schema: type[T],
    max_tokens: int = 4000,
) -> tuple[T, Usage]:
    """One schema-constrained call. `data` maps a label to untrusted text; each is wrapped
    in its own <document-NONCE> block (a fresh random tag per call, never guessable by the
    text) so instructions inside it are read as content, not commands. Labels are ours.
    Raises ValueError if the model refuses or the output does not validate."""
    model = TIERS[tier]
    tag = f"document-{secrets.token_hex(8)}"
    docs = "\n".join(
        f'<{tag} label="{label}">\n{_neutralise(text)}\n</{tag}>' for label, text in data.items()
    )
    prompt = f"{docs}\n\n{instruction}"
    if settings().llm_backend == "claude_cli":
        return _structured_cli(model, system, prompt, schema)
    resp = _client().messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
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


CLI_TIMEOUT_S = 300
# Rate limits and overload are transient: retried with backoff. A limit that outlasts the
# retries raises RateLimited, which callers treat as an infrastructure failure (the report
# fails and can be re-queued) rather than scoring the claim as unverifiable.
RETRY_WAITS_S = (15, 60, 180, 600)
_TRANSIENT = ("rate limit", "rate_limit", "429", "overloaded", "529", "usage limit",
              "session limit")


class RateLimited(RuntimeError):
    pass


def _structured_cli[T: BaseModel](
    model: str, system: str, prompt: str, schema: type[T],
) -> tuple[T, Usage]:
    """The same call through `claude -p`: our system prompt replaces the CLI's, every tool
    is disabled and the process runs in an empty directory, because the prompt carries
    untrusted filing text. The prompt goes on stdin, never argv (it would show in ps)."""
    for wait in (*RETRY_WAITS_S, None):
        with tempfile.TemporaryDirectory(prefix="ledgerline-cli-") as cwd:
            proc = subprocess.run(
                [settings().claude_bin, "-p", "--model", model, "--system-prompt", system,
                 "--strict-mcp-config", "--tools", "", "--output-format", "json",
                 "--json-schema", json.dumps(schema.model_json_schema())],
                input=prompt, capture_output=True, text=True, cwd=cwd, timeout=CLI_TIMEOUT_S,
                check=False,  # a non-zero exit still prints the JSON error we report below
            )
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out = {"is_error": True, "result": proc.stderr[-300:] or proc.stdout[-300:]}
        text = f"{out.get('result', '')} {out.get('subtype', '')} {proc.stderr[-300:]}".lower()
        if not (out.get("is_error") and any(t in text for t in _TRANSIENT)):
            break
        if wait is None:
            raise RateLimited(f"{model}: still rate-limited after {len(RETRY_WAITS_S)} retries")
        time.sleep(wait)
    if out.get("is_error") or out.get("structured_output") is None:
        raise ValueError(f"{model} returned no valid {schema.__name__} "
                         f"({out.get('subtype') or out.get('result', '')[:200]})")
    u = out.get("usage", {})
    return schema.model_validate(out["structured_output"]), Usage(
        model=model,
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read_tokens=u.get("cache_read_input_tokens", 0),
        cache_write_tokens=u.get("cache_creation_input_tokens", 0),
    )

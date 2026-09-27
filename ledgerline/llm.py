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
import logging
import os
import re
import secrets
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache

from pydantic import BaseModel, ValidationError

from ledgerline.config import settings

log = logging.getLogger(__name__)

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
            self.input_tokens * i
            + self.output_tokens * o
            + self.cache_read_tokens * r
            + self.cache_write_tokens * w
        ) / 1e6

    def __add__(self, other: "Usage") -> "Usage":
        if self.model and other.model and self.model != other.model:
            raise ValueError(f"cannot add {self.model} usage to {other.model} usage")
        return Usage(
            model=self.model or other.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


OFFLINE = Usage()


class LLMUnavailable(RuntimeError):
    """The model could not be reached (auth, network, timeout): an infrastructure failure the
    caller must fail and re-queue on, never score as a per-claim result. Messages are ours and
    fixed, never model output or CLI stderr."""


class RateLimited(LLMUnavailable):
    pass


def _bad_output(model: str, schema: type[BaseModel], detail: str = "") -> ValueError:
    """A refusal or unparseable answer. The message reaches the public API (tracing
    error_summary), so it carries no model output; that goes to the log."""
    return ValueError(
        f"{model} returned no valid {schema.__name__}" + (f" ({detail})" if detail else "")
    )


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
    tier: str,
    system: str,
    data: dict[str, str],
    instruction: str,
    schema: type[T],
    max_tokens: int = 4000,
) -> tuple[T, Usage]:
    """One schema-constrained call. `data` maps a label to untrusted text; each is wrapped
    in its own <document-NONCE> block (a fresh random tag per call, never guessable by the
    text) so instructions inside it are read as content, not commands. Labels are ours.
    Raises ValueError if the model refuses or the output does not validate, LLMUnavailable
    (RateLimited) if the model cannot be reached."""
    model = TIERS[tier]
    log.info("llm call", extra={"model": model, "schema": schema.__name__})
    tag = f"document-{secrets.token_hex(8)}"
    docs = "\n".join(
        f'<{tag} label="{label}">\n{_neutralise(text)}\n</{tag}>' for label, text in data.items()
    )
    prompt = f"{docs}\n\n{instruction}"
    if settings().llm_backend == "claude_cli":
        return _structured_cli(model, system, prompt, schema)
    import anthropic

    try:
        resp = _client().messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
    except anthropic.RateLimitError as e:  # the SDK has already retried
        raise RateLimited(f"{model}: rate-limited") from e
    except (anthropic.APIConnectionError, anthropic.CredentialsError) as e:  # incl. timeouts
        raise LLMUnavailable(f"{model}: unavailable ({type(e).__name__})") from e
    except anthropic.APIStatusError as e:  # auth, overload, server errors; a 400 is ours
        if e.status_code in (401, 403) or e.status_code >= 500:
            raise LLMUnavailable(f"{model}: unavailable (HTTP {e.status_code})") from e
        raise
    except ValidationError as e:
        log.warning("%s output failed %s validation: %s", model, schema.__name__, e)
        raise _bad_output(model, schema) from None
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise _bad_output(model, schema, str(resp.stop_reason))
    u = resp.usage
    return resp.parsed_output, Usage(
        model=model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        cache_write_tokens=u.cache_creation_input_tokens or 0,
    )


CLI_TIMEOUT_S = 300
# Rate limits, overload and timeouts are transient: retried with backoff. One that outlasts the
# retries raises RateLimited / LLMUnavailable, which callers treat as an infrastructure failure
# (the report fails and can be re-queued) rather than scoring the claim as unverifiable.
RETRY_WAITS_S = (15, 60, 180, 600)
# Known phrases and an HTTP status in its API-error context only: a bare "429" also occurs in
# ordinary text.
_TRANSIENT = re.compile(
    r"rate[ _]limit|overloaded|usage limit|session limit"
    r"|\b(?:api error|status(?: code)?|http)\W{0,3}(?:429|529)\b"
)
_AUTH = re.compile(
    r"invalid api key|/login|not logged in|authentication_error|oauth token|credit balance"
    r"|\b(?:api error|status(?: code)?|http)\W{0,3}(?:401|403)\b"
)
# The child gets only what it needs to find its own login; never our DB URL or API keys.
_CHILD_ENV = ("HOME", "PATH", "USER", "TMPDIR", "LANG")


def _cli_argv(model: str, system: str, schema: type[BaseModel]) -> list[str]:
    return [
        settings().claude_bin,
        "-p",
        "--model",
        model,
        "--system-prompt",
        system,
        "--strict-mcp-config",
        "--tools",
        "",
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema.model_json_schema()),
    ]


def _run_cli(argv: list[str], prompt: str) -> tuple[dict, str]:
    """One `claude -p` run: (its JSON result, error text for classification). A timeout or
    unparseable output reads as an error result; stderr is logged, never raised."""
    env = {k: os.environ[k] for k in _CHILD_ENV if k in os.environ}
    with tempfile.TemporaryDirectory(prefix="ledgerline-cli-") as cwd:
        try:
            proc = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=CLI_TIMEOUT_S,
                check=False,  # a non-zero exit still prints the JSON error we classify below
            )
        except subprocess.TimeoutExpired:  # err is exactly "timeout" only here
            return {"is_error": True, "subtype": "timeout"}, "timeout"
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = None
    if not isinstance(out, dict):
        out = {"is_error": True}
    if proc.stderr.strip() and (proc.returncode or out.get("is_error")):
        log.warning("claude CLI stderr (exit %s): %s", proc.returncode, proc.stderr[-500:])
    if out.get("is_error"):
        text = f"{out.get('result') or ''} {out.get('subtype') or ''} {proc.stderr[-300:]}"
        return out, text.lower()
    return out, ""


def _structured_cli[T: BaseModel](
    model: str,
    system: str,
    prompt: str,
    schema: type[T],
) -> tuple[T, Usage]:
    """The same call through `claude -p`: our system prompt replaces the CLI's, every tool,
    setting source, slash command and session file is off, and the process runs in an empty
    directory with a minimal env, because the prompt carries untrusted filing text. The
    prompt goes on stdin, never argv (it would show in ps)."""
    argv = _cli_argv(model, system, schema)
    for wait in (*RETRY_WAITS_S, None):
        out, err = _run_cli(argv, prompt)
        if _AUTH.search(err):
            raise LLMUnavailable(f"{model}: claude CLI is not authenticated")
        if not (err == "timeout" or _TRANSIENT.search(err)):
            break
        if wait is None:
            if err == "timeout":
                raise LLMUnavailable(f"{model}: timed out after {len(RETRY_WAITS_S)} retries")
            raise RateLimited(f"{model}: still rate-limited after {len(RETRY_WAITS_S)} retries")
        time.sleep(wait)
    if out.get("is_error") or out.get("structured_output") is None:
        log.warning("%s CLI error result: %s", model, str(out.get("result"))[:500])
        raise _bad_output(model, schema, str(out.get("subtype") or "no structured output"))
    try:
        parsed = schema.model_validate(out["structured_output"])
    except ValidationError as e:
        log.warning("%s output failed %s validation: %s", model, schema.__name__, e)
        raise _bad_output(model, schema) from None
    u = out.get("usage") or {}
    return parsed, Usage(
        model=model,
        input_tokens=u.get("input_tokens") or 0,
        output_tokens=u.get("output_tokens") or 0,
        cache_read_tokens=u.get("cache_read_input_tokens") or 0,
        cache_write_tokens=u.get("cache_creation_input_tokens") or 0,
    )

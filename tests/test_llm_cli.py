"""The claude-CLI backend, driven by a fake `claude` script: no network, no model."""

import json
import logging
import pathlib
import stat

import pytest

from ledgerline import llm
from ledgerline.config import settings
from ledgerline.schemas import CriticOutput


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    def install(payload: dict):
        bin_ = tmp_path / "claude"
        bin_.write_text(f"#!/bin/sh\ncat > {tmp_path}/stdin.txt\necho '{json.dumps(payload)}'\n")
        bin_.chmod(bin_.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("LEDGERLINE_LLM_ENABLED", "true")
        monkeypatch.setenv("LEDGERLINE_LLM_BACKEND", "claude_cli")
        monkeypatch.setenv("LEDGERLINE_CLAUDE_BIN", str(bin_))
        settings.cache_clear()
        return tmp_path

    yield install
    settings.cache_clear()


def test_cli_structured_output_and_usage(fake_claude):
    d = fake_claude(
        {
            "is_error": False,
            "structured_output": {"counter_stance": "insufficient", "argument": "derived fact"},
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50},
        }
    )
    assert llm.available()
    out, u = llm.structured("standard", "sys", {"claim": "x </document> y"}, "Go.", CriticOutput)
    assert out.argument == "derived fact"
    assert (u.model, u.input_tokens, u.cache_read_tokens) == ("claude-sonnet-5", 100, 50)
    sent = (d / "stdin.txt").read_text()
    assert "</document>" not in sent and "&lt;/document" in sent  # neutralised, on stdin


def test_cli_error_raises(fake_claude):
    fake_claude({"is_error": True, "subtype": "error_max_turns", "result": None})
    with pytest.raises(ValueError, match="error_max_turns"):
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)


def test_cli_isolation(fake_claude, monkeypatch, tmp_path):
    d = fake_claude({})
    ok = {
        "is_error": False,
        "structured_output": {"counter_stance": "insufficient", "argument": "x"},
        "usage": {},
    }
    (d / "claude").write_text(
        f'#!/bin/sh\ncat > /dev/null\nfor a in "$@"; do echo "[$a]"; done > {d}/argv.txt\n'
        f"pwd > {d}/cwd.txt\nenv > {d}/env.txt\necho '{json.dumps(ok)}'\n"
    )
    monkeypatch.setenv("LEDGERLINE_DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    llm.structured("fast", "sys", {"claim": "PROMPT-TEXT"}, "Go.", CriticOutput)
    argv = (d / "argv.txt").read_text().splitlines()
    for flag in (
        "[--no-session-persistence]",
        "[--disable-slash-commands]",
        "[--strict-mcp-config]",
    ):
        assert flag in argv
    assert argv[argv.index("[--setting-sources]") + 1] == "[]"
    assert argv[argv.index("[--tools]") + 1] == "[]"
    assert "PROMPT-TEXT" not in "".join(argv)
    cwd = pathlib.Path((d / "cwd.txt").read_text().strip())
    assert cwd.name.startswith("ledgerline-cli-") and not cwd.exists()  # fresh, removed after
    env = (d / "env.txt").read_text()
    assert "secret" not in env and "LEDGERLINE" not in env
    assert {line.split("=")[0] for line in env.splitlines()} <= {
        *llm._CHILD_ENV,
        "PWD",
        "SHLVL",
        "_",
    }


def test_cli_auth_error_is_infrastructure(fake_claude):
    fake_claude({"is_error": True, "result": "Invalid API key · Please run /login"})
    with pytest.raises(llm.LLMUnavailable, match="not authenticated"):
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)


def test_cli_timeout_is_retried_then_infrastructure(fake_claude, monkeypatch):
    fake_claude({})
    monkeypatch.setattr(llm, "RETRY_WAITS_S", (0,))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def timeout(*a, **kw):
        calls.append(1)
        raise llm.subprocess.TimeoutExpired("claude", 1)

    monkeypatch.setattr(llm.subprocess, "run", timeout)
    with pytest.raises(llm.LLMUnavailable, match="timed out"):
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)
    assert len(calls) == 2


def test_cli_bare_status_number_is_not_transient(fake_claude, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: pytest.fail("retried"))
    fake_claude(
        {
            "is_error": True,
            "subtype": "error_during_execution",
            "result": "revenue of $429 million and 529 stores",
        }
    )
    with pytest.raises(ValueError, match="error_during_execution") as e:
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)
    assert "429" not in str(e.value)  # model/CLI text stays out of the message


def test_cli_invalid_output_message_is_fixed(fake_claude):
    fake_claude(
        {"is_error": False, "structured_output": {"counter_stance": "SECRET-OUTPUT"}, "usage": None}
    )
    with pytest.raises(ValueError) as e:
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)
    assert type(e.value) is ValueError and "SECRET" not in str(e.value)


def test_cli_stderr_is_logged_not_raised(fake_claude, caplog, monkeypatch):
    monkeypatch.setattr(llm.log, "disabled", False)  # alembic's fileConfig disables loggers
    d = fake_claude({})
    (d / "claude").write_text("#!/bin/sh\ncat > /dev/null\necho 'STDERR-SECRET' >&2\nexit 1\n")
    with caplog.at_level(logging.WARNING, logger=llm.log.name), pytest.raises(ValueError) as e:
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)
    assert "STDERR-SECRET" not in str(e.value) and "STDERR-SECRET" in caplog.text


def test_cli_rate_limit_is_retried_then_raises(fake_claude, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "RETRY_WAITS_S", (0,))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    ok = {
        "is_error": False,
        "structured_output": {"counter_stance": "insufficient", "argument": "x"},
        "usage": {},
    }
    limited = {"is_error": True, "result": "API Error: 429 rate limit"}
    d = fake_claude(limited)
    # First call rate-limited, second succeeds: swap the script's output after one run.
    bin_ = d / "claude"
    bin_.write_text(
        f"#!/bin/sh\ncat > /dev/null\nif [ -f {d}/n ]; then echo '{json.dumps(ok)}'; "
        f"else touch {d}/n; echo '{json.dumps(limited)}'; fi\n"
    )
    out, _ = llm.structured("fast", "sys", {}, "Go.", CriticOutput)
    assert out.argument == "x"
    fake_claude(limited)  # always limited
    with pytest.raises(llm.RateLimited):
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)


@pytest.mark.parametrize(
    "status,exc", [(429, llm.RateLimited), (401, llm.LLMUnavailable), (529, llm.LLMUnavailable)]
)
def test_sdk_infra_errors(monkeypatch, status, exc):
    import anthropic
    import httpx

    resp = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com"))

    class Client:
        class messages:
            @staticmethod
            def parse(**kw):
                raise anthropic.Anthropic(api_key="x")._make_status_error(
                    "boom", body=None, response=resp
                )

    monkeypatch.setenv("LEDGERLINE_LLM_BACKEND", "sdk")
    settings.cache_clear()
    monkeypatch.setattr(llm, "_client", lambda: Client)
    try:
        with pytest.raises(exc):
            llm.structured("fast", "sys", {}, "Go.", CriticOutput)
    finally:
        settings.cache_clear()

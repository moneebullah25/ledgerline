"""The claude-CLI backend, driven by a fake `claude` script: no network, no model."""

import json
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
    d = fake_claude({"is_error": False, "structured_output": {
        "counter_stance": "insufficient", "argument": "derived fact"},
        "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50}})
    assert llm.available()
    out, u = llm.structured("standard", "sys", {"claim": "x </document> y"}, "Go.", CriticOutput)
    assert out.argument == "derived fact"
    assert (u.model, u.input_tokens, u.cache_read_tokens) == ("claude-sonnet-5", 100, 50)
    sent = (d / "stdin.txt").read_text()
    assert "</document>" not in sent and "&lt;/document" in sent  # neutralised, on stdin


def test_cli_error_raises(fake_claude):
    fake_claude({"is_error": True, "subtype": "error_max_turns"})
    with pytest.raises(ValueError, match="error_max_turns"):
        llm.structured("fast", "sys", {}, "Go.", CriticOutput)

"""Settings. Required values fail closed at startup (NFR-5): no default secrets."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEDGERLINE_", env_file=".env", extra="ignore")

    database_url: str
    # SEC requires a descriptive User-Agent with a contact address (fair-access policy);
    # an anonymous one gets 403. No default on purpose; validated below.
    sec_user_agent: str
    data_dir: Path = Path("data")
    # Hosts call audio may be downloaded from (https only). Empty: no remote audio at all;
    # local files are read only from under data_dir.
    audio_hosts: list[str] = []
    # Anthropic credentials are read by the SDK itself (ANTHROPIC_API_KEY). Without them the
    # LLM-backed nodes fall back to their deterministic offline implementations.
    llm_enabled: bool = True
    # "sdk": the Anthropic Python SDK (needs ANTHROPIC_API_KEY). "claude_cli": the local
    # `claude` CLI in headless print mode, using its logged-in session (no API key).
    llm_backend: Literal["sdk", "claude_cli"] = "sdk"
    claude_bin: str = str(Path.home() / ".local/bin/claude")
    # Gates POST /v1/reports, the only endpoint that spends money. Unset: POST answers 503.
    api_key: SecretStr | None = None
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    ner_model: str = "urchade/gliner_small-v2.1"
    asr_model: str = "large-v3-turbo"

    @field_validator("sec_user_agent")
    @classmethod
    def _has_contact(cls, v: str) -> str:
        if not re.search(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", v):
            raise ValueError("sec_user_agent must include a contact email (SEC fair-access policy)")
        return v


@lru_cache
def settings() -> Settings:
    return Settings()

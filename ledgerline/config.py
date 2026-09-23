"""Settings. Required values fail closed at startup (NFR-5): no default secrets."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEDGERLINE_", env_file=".env", extra="ignore")

    database_url: str
    # SEC requires a descriptive User-Agent with a contact address (fair-access policy);
    # an anonymous one gets 403. No default on purpose.
    sec_user_agent: str = Field(min_length=8)
    data_dir: Path = Path("data")
    # Anthropic credentials are read by the SDK itself (ANTHROPIC_API_KEY). Without them the
    # LLM-backed nodes fall back to their deterministic offline implementations.
    llm_enabled: bool = True
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    ner_model: str = "urchade/gliner_small-v2.1"
    asr_model: str = "large-v3-turbo"


@lru_cache
def settings() -> Settings:
    return Settings()

import pytest
from pydantic import ValidationError

from ledgerline.config import Settings


def test_settings_fail_closed_without_sec_user_agent(monkeypatch):
    # NFR-5: no default secret/identity; startup must refuse rather than run anonymously.
    monkeypatch.delenv("LEDGERLINE_SEC_USER_AGENT", raising=False)
    monkeypatch.setenv("LEDGERLINE_DATABASE_URL", "postgresql+psycopg://x@localhost/x")
    with pytest.raises(ValidationError, match="sec_user_agent"):
        Settings(_env_file=None)

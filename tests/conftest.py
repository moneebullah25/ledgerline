"""Shared fixtures.

Unit tests need nothing. Tests marked `db` run against a throwaway database
`<name>_test_<pid>_<rand>` on the same server as LEDGERLINE_DATABASE_URL, created and migrated
once per session and dropped at session end, so concurrent pytest runs never share one. Each
test runs inside a transaction that is rolled back. Tests marked `ml` skip when the [ml] extra
is not installed.
"""

import importlib.util
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("LEDGERLINE_SEC_USER_AGENT", "Ledgerline tests tests@example.com")
# Tests never call a live model: assigned, not defaulted, so a developer's shell can't opt in.
os.environ["LEDGERLINE_LLM_ENABLED"] = "false"
for _var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(_var, None)

ML_MODULES = ("sentence_transformers", "gliner", "faster_whisper")


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    """Re-applied per test: a test or import that set credentials can't leak them onward."""
    monkeypatch.setenv("LEDGERLINE_LLM_ENABLED", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def pytest_collection_modifyitems(config, items):
    missing = [m for m in ML_MODULES if importlib.util.find_spec(m) is None]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"[ml] extra not installed ({', '.join(missing)})")
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def db_url():
    base = os.environ.get("LEDGERLINE_DATABASE_URL")
    if not base:
        from dotenv import dotenv_values

        base = dotenv_values(ROOT / ".env").get("LEDGERLINE_DATABASE_URL")
    if not base:
        pytest.skip("no LEDGERLINE_DATABASE_URL")
    url = make_url(base)
    test_url = url.set(database=f"{url.database}_test_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    name = test_url.database
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except OperationalError as e:
        admin.dispose()
        pytest.skip(f"Postgres unreachable: {e.orig}")
    except DBAPIError as e:  # e.g. InsufficientPrivilege: role lacks CREATEDB
        admin.dispose()
        pytest.skip(f"cannot create test database: {e.orig}")

    from alembic import command
    from alembic.config import Config

    from ledgerline.config import settings

    rendered = test_url.render_as_string(hide_password=False)
    previous = os.environ.get("LEDGERLINE_DATABASE_URL")
    os.environ["LEDGERLINE_DATABASE_URL"] = rendered
    settings.cache_clear()
    try:
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "migrations"))
        command.upgrade(cfg, "head")
        yield rendered
    finally:
        if previous is None:
            os.environ.pop("LEDGERLINE_DATABASE_URL", None)
        else:
            os.environ["LEDGERLINE_DATABASE_URL"] = previous
        settings.cache_clear()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def db(db_url: str):
    """A Session bound to a connection whose outer transaction is always rolled back."""
    engine = create_engine(db_url)
    conn = engine.connect()
    outer = conn.begin()
    s = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        yield s
    finally:
        s.close()
        outer.rollback()
        conn.close()
        engine.dispose()

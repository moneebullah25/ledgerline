"""Shared fixtures.

Unit tests need nothing. Tests marked `db` run against a throwaway database
`<name>_test` on the same server as LEDGERLINE_DATABASE_URL, migrated to head once per
session; each test runs inside a transaction that is rolled back.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

os.environ.setdefault("LEDGERLINE_SEC_USER_AGENT", "Ledgerline tests tests@example.com")
os.environ.setdefault("LEDGERLINE_LLM_ENABLED", "false")  # tests never call a live model


@pytest.fixture(scope="session")
def db_url() -> str:
    base = os.environ.get("LEDGERLINE_DATABASE_URL")
    if not base:
        from dotenv import dotenv_values

        base = dotenv_values(".env").get("LEDGERLINE_DATABASE_URL")
    if not base:
        pytest.skip("no LEDGERLINE_DATABASE_URL")
    url = make_url(base)
    test_url = url.set(database=f"{url.database}_test")
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{test_url.database}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{test_url.database}"'))
    admin.dispose()

    from alembic import command
    from alembic.config import Config

    rendered = test_url.render_as_string(hide_password=False)
    os.environ["LEDGERLINE_DATABASE_URL"] = rendered
    from ledgerline.config import settings

    settings.cache_clear()
    command.upgrade(Config("alembic.ini"), "head")
    return rendered


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

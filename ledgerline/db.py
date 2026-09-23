from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ledgerline.config import settings


@lru_cache
def engine() -> Engine:
    return create_engine(settings().database_url, pool_pre_ping=True)


@contextmanager
def session() -> Iterator[Session]:
    """One transaction: commit on success, roll back on any exception."""
    with sessionmaker(engine(), expire_on_commit=False)() as s, s.begin():
        yield s

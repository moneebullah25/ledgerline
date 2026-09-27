from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ledgerline.config import settings

# Claims verified in parallel per report (pipeline.run_report), each on its own Session.
VERIFY_WORKERS = settings().verify_workers


@lru_cache
def engine() -> Engine:
    # One connection per verify worker plus the report's own Session; overflow covers the
    # API threadpool and bursts.
    return create_engine(settings().database_url, pool_pre_ping=True, pool_size=VERIFY_WORKERS + 1)


@contextmanager
def session() -> Iterator[Session]:
    """One transaction: commit on success, roll back on any exception."""
    with sessionmaker(engine(), expire_on_commit=False)() as s, s.begin():
        yield s

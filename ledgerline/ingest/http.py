"""Outbound HTTP and failure bookkeeping for every ingest flow.

sec_get is the only way to call SEC: EDGAR's fair-access policy allows 10 req/s per
client and blocks anonymous User-Agents. A 429 or 5xx is retried with backoff and never
recorded as permanent; any other 4xx is permanent (the resource is gone or forbidden).
"""

import threading
import time
from json import JSONDecodeError

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ledgerline.config import settings
from ledgerline.models import IngestFailure

SEC_MIN_INTERVAL = 0.1  # seconds between SEC requests, i.e. <= 10 req/s


class PermanentFetchError(Exception):
    def __init__(self, url: str, status: int):
        super().__init__(f"HTTP {status} for {url}")
        self.url, self.status = url, status


class TransientFetchError(Exception):
    pass


_client = httpx.Client(timeout=30, follow_redirects=True)


@retry(
    retry=retry_if_exception_type((TransientFetchError, httpx.TransportError)),
    wait=wait_exponential_jitter(initial=1, max=60),
    stop=stop_after_attempt(7),
    reraise=True,
)
def get(url: str, headers: dict[str, str], throttle: bool = False) -> bytes:
    if throttle:
        _throttle()
    r = _client.get(url, headers=headers)
    if r.status_code == 429 or r.status_code >= 500:
        raise TransientFetchError(f"HTTP {r.status_code} for {url}")
    if r.status_code >= 400:
        raise PermanentFetchError(url, r.status_code)
    return r.content


_lock = threading.Lock()
_last = 0.0


def _throttle() -> None:
    # Held across the sleep so concurrent callers queue up instead of bursting together.
    global _last
    with _lock:
        wait = _last + SEC_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()


def sec_get(url: str) -> bytes:
    # Throttle per attempt (inside the retry), so backoff retries also respect the limit.
    return get(url, {"User-Agent": settings().sec_user_agent}, throttle=True)


def error_class(exc: BaseException) -> str:
    if isinstance(exc, PermanentFetchError):
        return "http_4xx"
    if isinstance(exc, (TransientFetchError, httpx.TransportError)):
        return "http_transient_exhausted"
    if isinstance(exc, (KeyError, IndexError, TypeError)):
        return "schema_drift"  # an expected JSON key/shape is missing
    if isinstance(exc, LookupError):
        return "not_found"  # e.g. an earnings 8-K without an EX-99.1
    if isinstance(exc, (JSONDecodeError, ValueError)):
        return "parse"
    return type(exc).__name__[:60]


def record_failure(s: Session, flow: str, key: str, exc: BaseException | str,
                   cls: str | None = None) -> None:
    """Store a failure as a row. The same (flow, key, class) is stored once, so re-running
    a flow over a permanently broken input does not grow the table."""
    cls = cls or error_class(exc)  # type: ignore[arg-type]
    exists = s.scalar(select(IngestFailure.id).where(
        IngestFailure.flow == flow, IngestFailure.key == key, IngestFailure.error_class == cls))
    if exists is None:
        s.add(IngestFailure(flow=flow, key=key, error_class=cls, error=str(exc)[:2000]))
        s.flush()

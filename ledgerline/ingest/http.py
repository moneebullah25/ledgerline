"""Outbound HTTP and failure bookkeeping for every ingest flow.

sec_get is the only way to call SEC: EDGAR's fair-access policy allows 10 req/s per
client and blocks anonymous User-Agents. A 429 or 5xx is retried with backoff and never
recorded as permanent; any other 4xx is permanent (the resource is gone or forbidden).
Every download is https-only and size-capped, so a hostile or broken server can neither
downgrade the transport nor fill the disk.
"""

import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from json import JSONDecodeError

import httpx
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ledgerline.config import settings
from ledgerline.models import IngestFailure

SEC_MIN_INTERVAL = 0.1  # seconds between SEC requests, i.e. <= 10 req/s
MAX_BYTES = 200 * 2**20  # largest companyfacts payloads are tens of MB
MAX_RETRY_AFTER = 300.0  # seconds; a longer Retry-After is capped, not obeyed blindly


class PermanentFetchError(Exception):
    def __init__(self, url: str, status: int):
        super().__init__(f"HTTP {status} for {url}")
        self.url, self.status = url, status


class TransientFetchError(Exception):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class ResponseTooLarge(ValueError):
    pass


def _https_only(req: httpx.Request) -> None:
    # Checked on every hop BEFORE it is sent: checking the final URL after httpx followed the
    # redirects would already have sent a request to an http:// or internal target.
    if req.url.scheme != "https":
        raise ValueError(f"refusing non-https request {req.url}")


_client = httpx.Client(timeout=30, follow_redirects=True, event_hooks={"request": [_https_only]})
_backoff = wait_exponential_jitter(initial=1, max=60)


def _wait(rs: RetryCallState) -> float:
    after = getattr(rs.outcome.exception(), "retry_after", None)
    return min(after, MAX_RETRY_AFTER) if after is not None else _backoff(rs)


def _retry_after(value: str | None) -> float | None:
    """Retry-After is delta-seconds or an HTTP-date (RFC 9110 10.2.3)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError):
        return None


@retry(
    retry=retry_if_exception_type((TransientFetchError, httpx.TransportError)),
    wait=_wait,
    stop=stop_after_attempt(7),
    reraise=True,
)
def get(url: str, headers: dict[str, str], throttle: bool = False,
        max_bytes: int = MAX_BYTES) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL {url}")
    if throttle:
        _throttle()
    with _client.stream("GET", url, headers=headers) as r:
        if r.url.scheme != "https":
            raise ValueError(f"redirected to non-https URL {r.url}")
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientFetchError(f"HTTP {r.status_code} for {url}",
                                      _retry_after(r.headers.get("Retry-After")))
        if r.status_code >= 400:
            raise PermanentFetchError(url, r.status_code)
        if int(r.headers.get("Content-Length") or 0) > max_bytes:
            raise ResponseTooLarge(f"{url} is larger than {max_bytes} bytes")
        buf = bytearray()
        for chunk in r.iter_bytes():
            buf += chunk
            if len(buf) > max_bytes:
                raise ResponseTooLarge(f"{url} is larger than {max_bytes} bytes")
        return bytes(buf)


_lock = threading.Lock()
_last = 0.0


def _throttle() -> None:
    # ponytail: per-process limit; two backfills in parallel can reach 20 req/s. Run one
    # ingest process at a time, or move this to a shared (e.g. Postgres advisory) lock.
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
    if isinstance(exc, ResponseTooLarge):
        return "too_large"
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
    """Store a failure as a row. The same (flow, key, class) is stored once (uq_ingest_failure),
    so re-running a flow over a permanently broken input does not grow the table, and two
    workers recording the same failure at once do not collide."""
    cls = cls or error_class(exc)  # type: ignore[arg-type]
    s.execute(insert(IngestFailure).values(flow=flow, key=key, error_class=cls,
                                           error=str(exc)[:2000])
              .on_conflict_do_nothing(constraint="uq_ingest_failure"))

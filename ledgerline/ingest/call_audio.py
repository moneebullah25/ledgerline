"""Store call audio immutably, keyed by sha256 (FR-3). Transcription is audio.process_call."""

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlparse

import av
import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline.config import settings
from ledgerline.ingest.objects import put_object
from ledgerline.models import CallAudio


def _duration(path: Path) -> float | None:
    with av.open(str(path)) as c:
        return c.duration / av.time_base if c.duration else None


def ingest_call_audio(s: Session, path_or_url: str, *, kind: str, company_id: int | None,
                      event_date: date | None, license_note: str,
                      fiscal_year: int | None = None, fiscal_period: str | None = None) -> int:
    """Returns call_audio.id; the same bytes ingested twice return the existing row."""
    if urlparse(path_or_url).scheme in ("http", "https"):
        r = httpx.get(path_or_url, follow_redirects=True, timeout=300)
        r.raise_for_status()
        data, source_url, suffix = r.content, path_or_url, Path(urlparse(path_or_url).path).suffix
    else:
        p = Path(path_or_url)
        data, source_url, suffix = p.read_bytes(), p.resolve().as_uri(), p.suffix
    sha = hashlib.sha256(data).hexdigest()
    if (existing := s.scalar(select(CallAudio.id).where(CallAudio.sha256 == sha))) is not None:
        return existing
    _, rel = put_object(data, suffix.lower())
    row = CallAudio(
        company_id=company_id, kind=kind, event_date=event_date, fiscal_year=fiscal_year,
        fiscal_period=fiscal_period, storage_path=rel,
        duration_s=_duration(settings().data_dir / rel), source_url=source_url,
        fetched_at=datetime.now(UTC), sha256=sha, license_note=license_note)
    s.add(row)
    s.flush()
    return row.id

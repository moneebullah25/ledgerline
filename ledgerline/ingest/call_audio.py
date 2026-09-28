"""Store call audio immutably, keyed by sha256 (FR-3). Transcription is audio.process_call."""

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlparse

import av
from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline.config import settings
from ledgerline.ingest.http import get
from ledgerline.ingest.objects import put_object
from ledgerline.models import CallAudio

AUDIO_SUFFIXES = {".mp3", ".m4a", ".mp4", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm"}
MAX_AUDIO_BYTES = 2 * 2**30  # a multi-hour uncompressed WAV still fits


def _duration(path: Path) -> float | None:
    with av.open(str(path)) as c:
        return c.duration / av.time_base if c.duration else None


def ingest_call_audio(
    s: Session,
    path_or_url: str,
    *,
    kind: str,
    company_id: int | None,
    event_date: date | None,
    license_note: str,
    fiscal_year: int | None = None,
    fiscal_period: str | None = None,
) -> int:
    """Returns call_audio.id; the same bytes ingested twice return the existing row.

    Remote audio: https only, from a host in settings().audio_hosts. Local audio: only files
    under settings().data_dir. Either way the suffix must be a known audio format."""
    url = urlparse(path_or_url)
    suffix = Path(url.path if url.scheme else path_or_url).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise ValueError(f"not an audio file type ({suffix or 'no suffix'}): {path_or_url}")
    if url.scheme:
        allowed = frozenset(h.lower() for h in settings().audio_hosts)
        if url.scheme != "https" or url.hostname not in allowed:
            raise ValueError(
                f"audio URL not allowed (https + LEDGERLINE_AUDIO_HOSTS): {path_or_url}"
            )
        data, source_url = (
            get(path_or_url, {}, max_bytes=MAX_AUDIO_BYTES, allowed_hosts=allowed),
            path_or_url,
        )
    else:
        p = Path(path_or_url).resolve()
        if not p.is_relative_to(settings().data_dir.resolve()):
            raise ValueError(f"local audio must be under the data dir: {path_or_url}")
        if p.stat().st_size > MAX_AUDIO_BYTES:
            raise ValueError(f"audio file larger than {MAX_AUDIO_BYTES} bytes: {path_or_url}")
        data, source_url = p.read_bytes(), p.as_uri()
    sha = hashlib.sha256(data).hexdigest()
    if (existing := s.scalar(select(CallAudio.id).where(CallAudio.sha256 == sha))) is not None:
        return existing
    _, rel = put_object(data, suffix)
    row = CallAudio(
        company_id=company_id,
        kind=kind,
        event_date=event_date,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        storage_path=rel,
        duration_s=_duration(settings().data_dir / rel),
        source_url=source_url,
        fetched_at=datetime.now(UTC),
        sha256=sha,
        license_note=license_note,
    )
    s.add(row)
    s.flush()
    return row.id

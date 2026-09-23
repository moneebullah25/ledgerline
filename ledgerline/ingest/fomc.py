"""FOMC press conferences as the ASR eval set (FR-4).

Audio: the video the Board embeds on its own press-conference page (a Brightcove player on
federalreserve.gov). We read the page's video id, ask the player's public playback API for
the HLS manifest, and copy only the audio rendition with ffmpeg. No YouTube.
Reference: the Board's official transcript PDF, split into speaker turns.
"""

import json
import re
import subprocess
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline.config import settings
from ledgerline.ingest.call_audio import MAX_AUDIO_BYTES, ingest_call_audio
from ledgerline.ingest.http import get
from ledgerline.ingest.objects import put_object
from ledgerline.models import CallAudio

PAGE = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf{d}.htm"
PDF = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{d}.pdf"
LICENSE = ("Federal Reserve Board; 'Unless otherwise indicated, information on Board's website "
           "is in the public domain' (federalreserve.gov/disclaimer.htm). Audio track of the "
           "press-conference video embedded on the Board's page; transcript is the Board's PDF.")
_UA = {"User-Agent": "Mozilla/5.0 (Ledgerline research)"}
MAX_PAGE_BYTES = 50 * 2**20  # page, transcript PDF, player JSON, HLS playlists

_NOISE = re.compile(r"^(Page \d+ of \d+|FINAL|Transcript of .*|.*Press Conference|"
                    r"(January|February|March|April|May|June|July|August|September|October|"
                    r"November|December) \d{1,2}, \d{4})$")
# Turns start at a line like "CHAIR POWELL. ..." / "JO LING KENT. ..." (two+ caps words).
_SPEAKER = re.compile(r"^([A-Z][A-Z’'.-]+(?: [A-Z][A-Z’'.-]+)+)\.\s+")


def parse_transcript(text: str) -> list[dict]:
    """pdftotext output -> [{speaker, role, text}]. The chair is the executive, the most
    frequent other speaker is the moderator (operator), everyone else is a reporter."""
    turns: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _NOISE.match(line):
            continue
        if m := _SPEAKER.match(line):
            turns.append({"speaker": m[1], "text": line[m.end():]})
        elif turns:
            turns[-1]["text"] += " " + line
    others = Counter(t["speaker"] for t in turns if "CHAIR" not in t["speaker"]).most_common(1)
    moderator = others[0][0] if others else None
    for t in turns:
        t["role"] = ("executive" if "CHAIR" in t["speaker"]
                     else "operator" if t["speaker"] == moderator else "analyst")
    return turns


def _fetch(url: str, headers: dict[str, str] | None = None) -> bytes:
    return get(url, headers or _UA, max_bytes=MAX_PAGE_BYTES)


def _audio_manifest(page_html: str) -> str:
    video = re.search(r'data-video-id="(\d+)"', page_html)[1]
    account = re.search(r'data-account="(\d+)"', page_html)[1]
    cfg = json.loads(_fetch(
        f"https://players.brightcove.net/{account}/default_default/config.json"))
    meta = json.loads(_fetch(
        f"https://edge.api.brightcove.com/playback/v1/accounts/{account}/videos/{video}",
        {"Accept": f"application/json;pk={cfg['video_cloud']['policy_key']}"}))
    master = next(src["src"] for src in meta["sources"]
                  if src.get("type") == "application/x-mpegURL" and src["src"].startswith("https"))
    return audio_uri(master, _fetch(master).decode(errors="replace"))


def audio_uri(master: str, playlist: str) -> str:
    """The audio rendition is a separate playlist, so we never download the video frames.
    Its URI may be relative to the master playlist; the result must still be https."""
    uri = urljoin(master, re.search(r'#EXT-X-MEDIA:TYPE=AUDIO.*?URI="([^"]+)"', playlist)[1])
    if not uri.startswith("https://"):
        raise ValueError(f"audio rendition is not https: {uri}")
    return uri


def ingest_fomc_eval_set(s: Session, dates: list[str]) -> list[int]:
    """dates as YYYYMMDD. Skips dates already ingested (the audio download takes minutes)."""
    ids = []
    for d in dates:
        day = date(int(d[:4]), int(d[4:6]), int(d[6:]))
        existing = s.scalar(select(CallAudio.id).where(CallAudio.kind == "fomc",
                                                       CallAudio.event_date == day))
        if existing is not None:
            ids.append(existing)
            continue
        page = _fetch(PAGE.format(d=d)).decode(errors="replace")
        pdf = _fetch(PDF.format(d=d))
        # Under data_dir: ingest_call_audio reads local files from there only.
        settings().data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=settings().data_dir) as tmp:
            audio, pdf_path = Path(tmp) / f"fomc{d}.m4a", Path(tmp) / "t.pdf"
            # The playlist names further URLs ffmpeg follows: keep it on https, capped in size.
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error",
                            "-protocol_whitelist", "https,tls,tcp,crypto,hls",
                            "-i", _audio_manifest(page), "-vn", "-c:a", "copy",
                            "-fs", str(MAX_AUDIO_BYTES), str(audio)], check=True)
            pdf_path.write_bytes(pdf)
            text = subprocess.run(["pdftotext", str(pdf_path), "-"], check=True,
                                  capture_output=True, text=True).stdout
            call_id = ingest_call_audio(s, str(audio), kind="fomc", company_id=None,
                                        event_date=day, license_note=LICENSE)
        _, pdf_rel = put_object(pdf, ".pdf")
        ref = {"source_url": PDF.format(d=d), "pdf_path": pdf_rel, "turns": parse_transcript(text)}
        _, ref_rel = put_object(json.dumps(ref, indent=1).encode(), ".json")
        call = s.get_one(CallAudio, call_id)
        call.source_url, call.reference_transcript_path = PAGE.format(d=d), ref_rel
        s.flush()
        ids.append(call_id)
    return ids

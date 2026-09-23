import shutil
import subprocess
from datetime import UTC, date, datetime

import pytest

from ledgerline.audio.pipeline import build_turns, locate, segment_text
from ledgerline.models import CallAudio, Segment

SEGS = [
    {"start": 0.0, "end": 1.0, "words": [{"w": "Welcome", "s": 0.0, "e": 0.4},
                                         {"w": "everyone.", "s": 0.5, "e": 1.0}]},
    {"start": 1.2, "end": 2.0, "words": [{"w": "Ladies", "s": 1.2, "e": 1.5},
                                         {"w": "and", "s": 1.5, "e": 1.6},
                                         {"w": "gentlemen.", "s": 1.6, "e": 2.0}]},
    {"start": 2.5, "end": 3.1, "words": [{"w": "We", "s": 2.5, "e": 2.6},
                                         {"w": "reported", "s": 2.6, "e": 3.1}]},
]


def test_build_turns_merges_same_speaker():
    turns = build_turns(SEGS, [0, 0, 1])
    assert [t["text"] for t in turns] == ["Welcome everyone. Ladies and gentlemen.", "We reported"]
    assert [(t["t_start"], t["t_end"]) for t in turns] == [(0.0, 2.0), (2.5, 3.1)]
    assert [t["role"] for t in turns] == ["operator", "executive"]


def test_locate_word_times():
    words = SEGS[1]["words"]
    assert locate(words, 0, 6) == (1.2, 1.5)
    assert locate(words, 7, 20) == (1.5, 2.0)
    with pytest.raises(ValueError):
        locate(words, 6, 7)  # the space between words


@pytest.mark.db
def test_every_char_maps_to_segment_and_time(db):
    call = CallAudio(kind="fomc", storage_path="x", source_url="x", fetched_at=datetime.now(UTC),
                     sha256="0" * 64, license_note="test")
    db.add(call)
    db.flush()
    for i, t in enumerate(build_turns(SEGS, [0, 1, 1])):
        db.add(Segment(call_audio_id=call.id, idx=i, t_start=t["t_start"], t_end=t["t_end"],
                       speaker_label=t["speaker_label"], role=t["role"], section=t["section"],
                       text=t["text"], words=t["words"]))
    db.flush()
    text, spans = segment_text(db, call.id)
    segs = {s.id: s for s in db.query(Segment).filter_by(call_audio_id=call.id)}
    assert [segs[i].text for i, _, _ in spans] == text.split("\n")
    for pos, ch in enumerate(text):
        if ch.isspace():
            continue
        seg_id, start, _ = next(sp for sp in spans if sp[1] <= pos < sp[2])
        seg = segs[seg_id]
        t0, t1 = locate(seg.words, pos - start, pos - start + 1)
        assert seg.t_start <= t0 <= t1 <= seg.t_end


@pytest.mark.db
def test_ingest_is_idempotent(db, tmp_path, monkeypatch):
    from ledgerline.config import settings
    from ledgerline.ingest.call_audio import ingest_call_audio

    monkeypatch.setattr(settings(), "data_dir", tmp_path)
    wav = tmp_path / "a.wav"
    # 0.5 s of silence, 16 kHz mono PCM: a valid file without needing ffmpeg.
    import wave
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\0\0" * 8000)
    kw = {"kind": "earnings_call", "company_id": None, "event_date": date(2024, 1, 1),
          "license_note": "test"}
    first = ingest_call_audio(db, str(wav), **kw)
    assert ingest_call_audio(db, str(wav), **kw) == first
    assert db.query(CallAudio).filter_by(sha256=db.get(CallAudio, first).sha256).count() == 1
    row = db.get(CallAudio, first)
    assert row.duration_s == pytest.approx(0.5, abs=0.05)
    assert (tmp_path / row.storage_path).exists()


@pytest.mark.ml
@pytest.mark.skipif(not (shutil.which("say") and shutil.which("ffmpeg")), reason="needs say+ffmpeg")
def test_vad_asr_smoke(tmp_path):
    from ledgerline.audio.pipeline import load_audio, speech_spans, transcribe

    aiff, wav = tmp_path / "x.aiff", tmp_path / "x.wav"
    subprocess.run(["say", "-o", str(aiff), "Revenue grew ten percent this quarter."], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(aiff), "-ar", "16000",
                    "-ac", "1", str(wav)], check=True)
    audio = load_audio(str(wav))
    assert speech_spans(audio)
    words = [w for seg in transcribe(audio, "base") for w in seg["words"]]
    assert "revenue" in " ".join(w["w"] for w in words).lower()
    assert all(w["s"] <= w["e"] for w in words)

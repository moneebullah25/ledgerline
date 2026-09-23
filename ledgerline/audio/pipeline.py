"""VAD -> ASR -> speaker clusters -> roles -> sections -> segment rows (FR-10..14).

Model imports are deferred so importing this module (and `ledgerline.audio`) stays cheap
for callers that only need `segment_text`.
"""

import os
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline.audio.roles import assign_roles, sections
from ledgerline.config import settings
from ledgerline.models import CallAudio, Segment

SR = 16_000
VAD = {"min_speech_duration_ms": 250, "speech_pad_ms": 200}  # FR-10
# Without a prompt large-v3-turbo often emits lowercase text with no punctuation, which
# breaks the sentence-level role cues. A neutral punctuated prompt fixes the style.
PROMPT = "Good afternoon, and thank you for joining us."
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
# Cosine distance under which two segments count as the same voice. WavLM-SV vectors sit
# close together (every voice in the FOMC excerpt was within 0.25), so the cut is tight.
# Swept 0.04-0.6 on the first 15 min of the 2024-09-18 FOMC press conference: tuned on the
# eval set itself, one event, so treat it as a starting point for earnings calls.
SPEAKER_THRESHOLD = float(os.environ.get("LEDGERLINE_SPEAKER_THRESHOLD", "0.15"))


@dataclass
class Result:
    turns: list[dict]  # {t_start, t_end, speaker_label, role, section, text, words}
    duration_s: float
    speech_s: float
    timings: dict[str, float]


def load_audio(path: str, clip_s: float | None = None) -> np.ndarray:
    from faster_whisper import decode_audio

    audio = decode_audio(path, sampling_rate=SR)
    return audio[: int(clip_s * SR)] if clip_s else audio


def speech_spans(audio: np.ndarray) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return [(c["start"] / SR, c["end"] / SR)
            for c in get_speech_timestamps(audio, VadOptions(**VAD))]


@lru_cache(maxsize=2)
def _whisper(name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(audio: np.ndarray, model: str, vad: bool = True) -> list[dict]:
    """Whisper segments as {start, end, words: [{w, s, e}]}; VAD runs inside faster-whisper
    with the same options, and timestamps come back on the original timeline."""
    segs, _ = _whisper(model).transcribe(
        audio, language="en", word_timestamps=True, vad_filter=vad,
        vad_parameters=VAD if vad else None, initial_prompt=PROMPT)
    out = []
    for seg in segs:
        words = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
                 for w in seg.words or [] if w.word.strip()]
        if words:
            out.append({"start": words[0]["s"], "end": words[-1]["e"], "words": words})
    return out


@lru_cache(maxsize=1)
def _speaker_model():
    from transformers import AutoFeatureExtractor, WavLMForXVector

    return (AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL),
            WavLMForXVector.from_pretrained(SPEAKER_MODEL).eval())


def embed_speakers(audio: np.ndarray, segs: list[dict]) -> np.ndarray:
    """One x-vector per whisper segment, from a 1.5-6 s window around its centre: sub-second
    clips give unstable embeddings, and long ones cost quadratic attention for no gain."""
    import torch

    fx, model = _speaker_model()
    out = []
    for seg in segs:
        mid = (seg["start"] + seg["end"]) / 2
        half = min(max((seg["end"] - seg["start"]) / 2, 0.75), 3.0)
        clip = audio[max(0, int((mid - half) * SR)): int((mid + half) * SR)]
        with torch.no_grad():
            emb = model(**fx(clip, sampling_rate=SR, return_tensors="pt")).embeddings[0]
        out.append(torch.nn.functional.normalize(emb, dim=-1).numpy())
    return np.stack(out)


def cluster(emb: np.ndarray, threshold: float = SPEAKER_THRESHOLD) -> list[int]:
    if len(emb) < 2:
        return [0] * len(emb)
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=threshold).fit_predict(emb).tolist()


def build_turns(segs: list[dict], labels: list[int]) -> list[dict]:
    """Merge consecutive whisper segments with the same speaker into one turn, then label
    roles and sections over the turn texts."""
    turns: list[dict] = []
    for seg, lab in zip(segs, labels, strict=True):
        if turns and turns[-1]["speaker_label"] == f"spk{lab}":
            turns[-1]["words"] += seg["words"]
        else:
            turns.append({"speaker_label": f"spk{lab}", "words": list(seg["words"])})
    for t in turns:
        # text is exactly the words joined by one space, so a char offset maps to a word.
        t["text"] = " ".join(w["w"] for w in t["words"])
        t["t_start"], t["t_end"] = t["words"][0]["s"], t["words"][-1]["e"]
    texts = [t["text"] for t in turns]
    for t, role, sec in zip(turns, assign_roles(texts, [t["speaker_label"] for t in turns]),
                            sections(texts), strict=True):
        t["role"], t["section"] = role, sec
    return turns


def run(path: str, *, model: str | None = None, clip_s: float | None = None) -> Result:
    """The whole pipeline on one file, without the database. clip_s keeps only the first
    clip_s seconds (benchmark excerpts). Timings exclude model loading."""
    model = model or settings().asr_model  # LEDGERLINE_ASR_MODEL overrides via settings
    _whisper(model), _speaker_model()
    tm: dict[str, float] = {}
    t0 = time.perf_counter()
    audio = load_audio(path, clip_s)
    spans = speech_spans(audio)
    tm["decode_vad_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    segs = transcribe(audio, model)
    tm["asr_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels = cluster(embed_speakers(audio, segs)) if segs else []
    tm["speaker_s"] = time.perf_counter() - t0
    return Result(build_turns(segs, labels), len(audio) / SR,
                  sum(e - s for s, e in spans), tm)


def process_call(s: Session, call_audio_id: int, *, model: str | None = None) -> int:
    """Transcribe one stored call into segment rows. Idempotent: a call that already has
    segments is left alone (claims may reference them)."""
    existing = s.scalars(select(Segment.id).where(Segment.call_audio_id == call_audio_id)).all()
    if existing:
        return len(existing)
    call = s.get_one(CallAudio, call_audio_id)
    res = run(str(settings().data_dir / call.storage_path), model=model)
    call.speech_s = res.speech_s
    s.add_all(Segment(call_audio_id=call_audio_id, idx=i, t_start=t["t_start"],
                      t_end=t["t_end"], speaker_label=t["speaker_label"], role=t["role"],
                      section=t["section"], text=t["text"], words=t["words"])
              for i, t in enumerate(res.turns))
    s.flush()
    return len(res.turns)


def segment_text(s: Session, call_audio_id: int) -> tuple[str, list[tuple[int, int, int]]]:
    """Segments joined by newlines, plus (segment_id, char_start, char_end) per segment."""
    rows = s.execute(select(Segment.id, Segment.text).where(Segment.call_audio_id == call_audio_id)
                     .order_by(Segment.idx)).all()
    spans, pos = [], 0
    for seg_id, text in rows:
        spans.append((seg_id, pos, pos + len(text)))
        pos += len(text) + 1
    return "\n".join(t for _, t in rows), spans


def locate(words: list[dict], char_start: int, char_end: int) -> tuple[float, float]:
    """Time range of the words overlapping [char_start, char_end) of a segment's text."""
    hits, pos = [], 0
    for w in words:
        end = pos + len(w["w"])
        if pos < char_end and end > char_start:
            hits.append(w)
        pos = end + 1
    if not hits:
        raise ValueError(f"span {char_start}:{char_end} covers no word")
    return hits[0]["s"], hits[-1]["e"]

"""L1 ASR benchmark on a real FOMC press conference (FR-4).

    python -m ledgerline.evals.wer [--date 20240918] [--model small] [--minutes 15]

Runs the audio pipeline on the first N minutes, aligns the output to the Board's official
transcript, and reports WER, real-time factor, VAD reduction, a VAD speech-recall proxy,
and role/section accuracy. Writes one eval_run row (layer L1).

Reference caveat: the official transcript is lightly edited (false starts and repeated
words are sometimes dropped), so WER against it overstates true ASR error a little.
"""

import argparse
import json
import subprocess
from collections import Counter
from difflib import SequenceMatcher

import jiwer

from ledgerline.audio import pipeline
from ledgerline.config import settings
from ledgerline.db import session
from ledgerline.ingest.fomc import ingest_fomc_eval_set
from ledgerline.ingest.objects import read_object
from ledgerline.models import CallAudio, EvalRun


def normalize(text: str) -> list[str]:
    """Whisper's English normalizer (lowercase, no punctuation, spelled numbers -> digits,
    bracketed annotations like [Laughter] removed), applied identically to both sides."""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer({})(text.replace("½", " half ").replace("—", " ")).split()


def _tokens(turns: list[dict]) -> tuple[list[str], list[int]]:
    """Normalized tokens of all turns plus, per token, the index of its turn."""
    toks, owner = [], []
    for i, t in enumerate(turns):
        words = normalize(t["text"])
        toks += words
        owner += [i] * len(words)
    return toks, owner


ANCHOR = 5  # words; shorter matches ("thank you") recur all over a transcript


def align(ref: list[str], hyp: list[str]) -> tuple[int, dict[int, int]]:
    """(end of the reference portion the hypothesis covers, hyp index -> ref index for
    matched tokens). The cut is the end of the last match of >= ANCHOR words: a stray short
    match of the excerpt's final words far ahead in the reference would otherwise count
    the whole gap as deletions. Trailing deletions at the excerpt boundary are not counted."""
    blocks = [b for b in SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks()
              if b.size]
    anchors = [b for b in blocks if b.size >= ANCHOR] or blocks
    cut = max((b.a + b.size for b in anchors), default=0)
    pairs = {b.b + k: b.a + k for b in blocks if b.a < cut for k in range(b.size)}
    return cut, pairs


def label_accuracy(hyp_turns, hyp_owner, ref_turns, ref_owner, pairs, key) -> dict:
    """turn_acc: a hypothesis turn is right if its label equals the majority reference label
    of its aligned words. word_acc: per aligned word, hypothesis label vs reference label, so
    a turn that swallows another speaker's words is penalised for them."""
    gold: dict[int, Counter] = {}
    word_ok = 0
    for h, r in pairs.items():
        label = ref_turns[ref_owner[r]][key]
        gold.setdefault(hyp_owner[h], Counter())[label] += 1
        word_ok += hyp_turns[hyp_owner[h]][key] == label
    turn_ok = [hyp_turns[i][key] == c.most_common(1)[0][0] for i, c in gold.items()]
    return {"turns": len(turn_ok),
            "turn_acc": round(sum(turn_ok) / len(turn_ok), 4) if turn_ok else None,
            "word_acc": round(word_ok / len(pairs), 4) if pairs else None}


def vad_recall_proxy(audio_path: str, model: str, minutes: float) -> float:
    """Share of words whisper finds WITHOUT VAD whose midpoint falls inside a VAD speech
    span. A proxy: the reference has no timestamps, so no-VAD ASR stands in for ground truth."""
    audio = pipeline.load_audio(audio_path, minutes * 60)
    spans = pipeline.speech_spans(audio)
    mids = [(w["s"] + w["e"]) / 2 for seg in pipeline.transcribe(audio, model, vad=False)
            for w in seg["words"]]
    if not mids:
        raise ValueError("no words transcribed without VAD; cannot estimate recall")
    return sum(any(s <= m <= e for s, e in spans) for m in mids) / len(mids)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.wer")
    p.add_argument("--date", default="20240918")
    p.add_argument("--model", default=None, help="faster-whisper model; default settings().asr_model")
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--vad-recall-minutes", type=float, default=3.0, help="0 skips the proxy")
    a = p.parse_args(argv)
    model = a.model or settings().asr_model

    with session() as s:
        call_id = ingest_fomc_eval_set(s, [a.date])[0]
        call = s.get_one(CallAudio, call_id)
        audio_path = str(settings().data_dir / call.storage_path)
        ref_turns = json.loads(read_object(call.reference_transcript_path))["turns"]
        source, full_s = call.source_url, call.duration_s

    for i, t in enumerate(ref_turns):
        t["section"] = "prepared_remarks" if i == 0 else "qa"  # chair's statement is turn 0
    res = pipeline.run(audio_path, model=model, clip_s=a.minutes * 60)
    ref, ref_owner = _tokens(ref_turns)
    hyp, hyp_owner = _tokens(res.turns)
    cut, pairs = align(ref, hyp)
    wer = jiwer.wer(" ".join(ref[:cut]), " ".join(hyp))
    run_s = sum(res.timings.values())
    metrics = {
        "asr_wer": round(wer, 4),
        "rtf": round(run_s / res.duration_s, 4),
        "asr_rtf": round(res.timings["asr_s"] / res.duration_s, 4),
        "vad_reduction": round(1 - res.speech_s / res.duration_s, 4),
        "role_acc": label_accuracy(res.turns, hyp_owner, ref_turns, ref_owner, pairs, "role"),
        "section_acc": label_accuracy(res.turns, hyp_owner, ref_turns, ref_owner, pairs,
                                      "section"),
    }
    if a.vad_recall_minutes:
        metrics["vad_speech_recall_proxy"] = round(
            vad_recall_proxy(audio_path, model, a.vad_recall_minutes), 4)
    details = {
        "model": model, "compute_type": "int8", "device": "cpu",
        "audio_source": source, "call_audio_id": call_id, "synthetic": False,
        "evaluated_s": round(res.duration_s, 1), "full_duration_s": full_s,
        "speech_s": round(res.speech_s, 1), "ref_words_aligned": cut, "hyp_words": len(hyp),
        "segments": len(res.turns), "speakers": len({t["speaker_label"] for t in res.turns}),
        "speaker_threshold": pipeline.SPEAKER_THRESHOLD, "timings_s": res.timings,
        "vad_recall_minutes": a.vad_recall_minutes,
        "reference": "Board transcript PDF (lightly edited); tokens after the last aligned "
                     "match are excluded",
    }
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=False).stdout.strip()
    with session() as s:
        s.add(EvalRun(git_sha=sha or None, layer="L1", name="wer", metrics=metrics, details=details))
    print(json.dumps({"metrics": metrics, "details": details}, indent=1))
    return metrics


if __name__ == "__main__":
    main()

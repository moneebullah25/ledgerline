# Ledgerline

Checks what management says in quarterly earnings releases against what the company filed
with the SEC, and produces a severity-ranked report where every number shown traces to an
XBRL fact, a filing passage, or a value computed from them.

Built 23-24 September 2026 as the capstone of `DevBooks/LEDGERLINE_ROADMAP.md`. The original
PRD file was missing; `docs/PRD.md` is a reconstruction from the roadmap.

## Results, measured (offline)

No Anthropic API key was configured, so every LLM-backed step ran its deterministic offline
implementation. The Claude path (Haiku / Sonnet 5 / Opus 5 tiering, `ledgerline/llm.py`) is
written and unit-tested with a stubbed model but has never run live.

| Metric (target) | Measured | Source |
|---|---|---|
| Contradiction precision, L2 — golden/frozen (>= 0.70) | 0.412 current extraction; 0.226 frozen snapshot. **Not met on golden set.** | eval_run 13 / 12 |
| Contradiction precision, L2 — held-out (>= 0.70) | **0.826** [0.612, 0.950] 95% CI, n=110. **Met.** Offline deterministic graph only. | eval_run 15 |
| Citation integrity (100%) | 0 violations in a 733-claim run; enforced as a hard check on every verdict | `schemas.uncited_numbers` |
| Claims per release (>= 30) | 30.5 mean over 120 releases (median below 30) | s4-rules-3 run |
| ASR WER, FOMC benchmark (<= 12%) | 6.7% (first 15 min), 9.6% (full hour) | eval_run 6 |
| Retrieval Recall@10 (>= 0.80) | 0.794 (hybrid + rerank). Missed by 0.006. | eval_run 3 |
| NER span F1 (>= 0.75) | 0.804 exact, 0.906 partial | eval_run 4 |
| Forecast MASE vs seasonal naive (< 1.0) | 0.840 mean over 41 series | eval_run 1 |
| Backtest precision on flags, L4 (>= 0.55) | 0.75, but on only 4 flags: no real signal | eval_run 14 |
| Judge-human agreement, L3 (>= 0.60) | not measured: no live judge, no human labels | eval_run 9 |
| Report latency (<= 20 min) | 34-40 s per company-quarter | capstone runs |
| API p95 on a cached report (<= 300 ms) | 11.1 ms in-process | perf test |

The golden set (200 real claims) was labelled by an AI agent before the system ran on it,
not by a human. Most false contradictions come from extraction normalization errors (segment
figures read as company totals, period stamping), not from the XBRL comparison itself.

## Run it

```bash
cp .env.example .env        # set POSTGRES_PASSWORD, LEDGERLINE_SEC_USER_AGENT (with a contact email)
docker compose up -d db
uv sync --extra ml
uv run alembic upgrade head
uv run python -m ledgerline.ingest backfill --ticker AAPL   # filings, XBRL facts, prices
uv run python -m ledgerline.store index --all                # chunks + embeddings
uv run python -m ledgerline.extract                          # claims from 8-K earnings releases
uv run python -m ledgerline.pipeline --ticker AAPL --fy 2026 --fp Q3   # the cited report
uv run uvicorn ledgerline.api.main:app                       # §15 API; POST needs LEDGERLINE_API_KEY
make test                                                    # ruff + pytest (no ml/network/perf)
```

`make up` / `make backfill` run the same steps in containers. The API image was built and
served `/healthz`; the ML worker image (with ffmpeg and poppler) was not built in this session.

## Layout

| Path | What |
|---|---|
| `ledgerline/ingest/` | EDGAR filings and 8-K exhibit 99.1, XBRL companyfacts, Yahoo prices, FOMC audio, Prefect flows |
| `ledgerline/store/` | Section-aware chunking, pgvector + full-text hybrid search, typed XBRL queries |
| `ledgerline/audio/` | Silero VAD, faster-whisper int8, speaker roles, section split |
| `ledgerline/extract/` | GLiNER + rules claim extraction, period resolution, guidance-to-actual linking |
| `ledgerline/graph/` | LangGraph verification: router, four evidence agents, refusal, critic, adjudicator |
| `ledgerline/report/`, `api/`, `pipeline.py`, `worker.py` | Severity-ranked report, REST API, Postgres job queue |
| `ledgerline/evals/` | L1-L4 evaluation, golden set and fixture, CI regression gate |
| `docs/` | PRD (reconstructed), module contracts, four ADRs, both tester-panel rounds |

The design record, both review rounds and an interview bank are in
`DevBooks/ledgerline-book.pdf`.

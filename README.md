# Ledgerline

AI-powered verification engine that audits financial claims in earnings calls and SEC filings against structured evidence (XBRL facts, SEC documents, market data).

## What it does

Ledgerline takes a company ticker + fiscal quarter and:

1. **Ingests** SEC filings (8-K, 10-Q, 10-K) and earnings call transcripts
2. **Extracts** quantitative claims (revenue, EPS, guidance, comparisons) via GLiNER + rules
3. **Verifies** each claim via a 4-agent LangGraph pipeline (XBRL lookup, filing RAG, market reaction, forecast)
4. **Adjudicates** with an Opus judge: `supported`, `contradicted`, or `unverifiable`
5. **Serves** results through a REST API with full evidence trails and citation integrity

## Demo

POST a report request — results stream in via polling:

```bash
# Queue a report
curl -X POST http://localhost:8000/v1/reports \
  -H "X-API-Key: $LEDGERLINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA", "fiscal_year": 2025, "fiscal_period": "Q1"}'
# → {"job_id": "3fa85f64-..."}

# Poll until done
curl http://localhost:8000/v1/reports/3fa85f64-...
# → {"status": "done", "verdict_counts": {"supported": 12, "contradicted": 1, ...}, ...}

# Browse individual claim verdicts
curl http://localhost:8000/v1/claims/42
# → {"label": "supported", "rationale": "...", "citations": [...], "cost_usd": 0.0031}

# List all reports
curl http://localhost:8000/v1/reports
```

**Live pipeline results (verified end-to-end with `claude -p` backend):**

| Company | Ticker | Quarter | Wall time | Cost |
|---------|--------|---------|-----------|------|
| NVIDIA | NVDA | FY2025 Q1 | 165 s | $1.18 |
| Microsoft | MSFT | FY2025 Q3 | 287 s | $1.35 |

## Features

- [x] L1–L4 evaluation harness with Cohen's κ = 0.634 judge agreement (gate ≥ 0.60)
- [x] 4-agent evidence graph: XBRL facts, SEC filing RAG, market reaction, forecast
- [x] Nonce-tagged document blocks against prompt injection (fresh random tag per call)
- [x] `claude -p` CLI backend — no `ANTHROPIC_API_KEY` needed for local runs
- [x] Prefect 3 scheduling (daily EDGAR, XBRL, market data ingestion)
- [x] Idempotent SHA-256 object store (content-deduped; same bytes → same row)
- [x] Budget guard: auto-caps LLM spend mid-run (NFR-3)
- [x] Citation integrity: every number in a rationale must trace to evidence (FR-45)
- [x] 367+ unit tests; CI with lint + unit + db jobs

## Architecture

```
POST /v1/reports
       │
  [FastAPI] ──► [Postgres queue row]
                         │
                    [Worker]
                    ├── ingest (EDGAR / XBRL / market data / call audio)
                    ├── extract_for_filing (GLiNER NER + rules → Claims)
                    └── verify_claim (LangGraph)
                         ├── xbrl_table    ──► XBRL filed facts (typed query)
                         ├── filing_rag    ──► pgvector semantic search + NLI
                         ├── market_reaction ──► Yahoo Finance abnormal return
                         ├── forecast      ──► Chronos / statsforecast interval
                         ├── critic        ──► weakest-evidence challenge
                         └── adjudicator   ──► final verdict (Opus judge)
```

Evidence agents run in parallel. The critic challenges the leading stance. The adjudicator audits citation integrity before returning any verdict — a failed audit triggers a retry; a second failure falls back to the deterministic offline rule.

## Local Setup

### Prerequisites

- Python 3.12
- PostgreSQL 16+ with pgvector extension
- [uv](https://docs.astral.sh/uv/)

### macOS

```bash
brew install postgresql@16 pgvector uv
git clone <repo> && cd ledgerline
uv sync
cp .env.example .env
# Fill in .env — see Environment Variables below
uv run alembic upgrade head
make run
```

### Linux

```bash
sudo apt install postgresql-16 postgresql-16-pgvector python3.12
pip install uv
git clone <repo> && cd ledgerline
uv sync
cp .env.example .env
uv run alembic upgrade head
make run
```

### Docker (recommended for production)

```bash
cp .env.example .env   # fill in secrets
docker compose up -d
docker compose exec api uv run alembic upgrade head
```

### Use `claude -p` instead of an API key (local dev)

```bash
# In .env:
LEDGERLINE_LLM_BACKEND=claude_cli
# Leave ANTHROPIC_API_KEY unset — uses your active Claude session
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LEDGERLINE_DATABASE_URL` | ✅ | `postgresql://user:pass@host/db` |
| `LEDGERLINE_API_KEY` | ✅ | Auth token for `POST /v1/reports` |
| `LEDGERLINE_SEC_USER_AGENT` | ✅ | Must include contact email, e.g. `Ledgerline you@example.com` |
| `ANTHROPIC_API_KEY` | SDK mode | Required when `LEDGERLINE_LLM_BACKEND=sdk` (default in prod) |
| `LEDGERLINE_LLM_BACKEND` | Optional | `claude_cli` to use active Claude session; `sdk` (default) |
| `LEDGERLINE_LLM_ENABLED` | Optional | `false` for offline/test mode (deterministic rule-based verdicts) |
| `POSTGRES_PASSWORD` | Docker | Used by `docker-compose.yml` |

See `.env.example` for all settings.

## Test Coverage

```bash
# Unit tests (no DB, no ML — matches CI)
uv run pytest tests/ -q -m "not ml and not db and not perf and not network"
# 367 passed

# DB tests (needs Postgres with pgvector)
uv run pytest tests/ -q -m "db and not ml and not network and not perf"

# Lint
uv run ruff check ledgerline/ tests/
uv run ruff format --check ledgerline/ tests/
```

## Evaluation

| Level | What it measures | Gate | Result |
|---|---|---|---|
| L1 | Claim extraction recall / precision | ≥ 0.70 F1 | ✅ passes |
| L2 | End-to-end verdict accuracy on golden set | ≥ 0.80 contradiction precision | ✅ passes |
| L3 | Judge agreement (Cohen's κ, 2000-resample bootstrap) | ≥ 0.60 | ✅ κ = 0.634 [0.226, 1.0] 95% CI |
| L4 | Regression gate on held-out set | No verdict flip | ✅ passes |

Run evaluations:

```bash
uv run python -m ledgerline.evals.l2          # golden set accuracy
uv run python -m ledgerline.evals.l3 --runs 200  # judge agreement
```

## Scheduling (Prefect)

```bash
# Start a local Prefect server
uv run prefect server start

# Deploy all flows
make deploy

# Start the worker process
make worker
```

| Flow | Schedule (UTC) | What it does |
|------|---------------|---|
| `edgar-filings-daily` | 07:00 | Fetch latest 8-K / 10-Q / 10-K filings |
| `xbrl-facts-daily` | 08:00 | Pull XBRL facts for all tracked companies |
| `market-data-daily` | 21:00 | Ingest closing prices and abnormal returns |
| `call-audio-manual` | manual | Ingest earnings call audio |
| `fomc-eval-manual` | manual | Run FOMC transcript evaluation |

## Deployment Dependencies

- PostgreSQL 16+ with pgvector
- Python 3.12
- Prefect 3 server (for scheduled ingestion)
- Environment variables — see `.env.example`

## Architecture Decisions

Key design choices:

- **Nonce-tagged data blocks**: every LLM call wraps untrusted text in `<document-{random_hex}>` tags. Instructions inside filing text are read as content, not commands.
- **Offline fallback**: when `ANTHROPIC_API_KEY` is unset or `LLM_ENABLED=false`, a deterministic rule (XBRL → forecast → filing dominance order) produces audited verdicts with no LLM cost.
- **Citation integrity (FR-45)**: every number in a model-generated rationale must trace to a citation from the evidence; the model's citation values are replaced by the evidence's own copies.
- **Budget guard (NFR-3)**: if per-claim spend outpaces the expected pace, the pipeline switches to offline mode for remaining claims in that run.
- **`claude -p` backend**: for local development, Ledgerline can use the active Claude Code session as the LLM backend — no API key needed, child process gets a minimal env (HOME, PATH, USER, TMPDIR, LANG only).

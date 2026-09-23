# Module contracts

The subsystems were built in parallel against these signatures. Anything not listed here
is private to its package. Shared, already-written files (do not rewrite; extend only if a
contract below requires it): `models.py` (the §9 schema, migrated), `schemas.py` (§22 contracts
and the citation-integrity check), `concepts.py` (canonical metrics, tag aliases, universe),
`llm.py` (the only Claude entry point), `tracing.py`, `config.py`, `db.py`.

## S1 Ingest — `ledgerline/ingest/`

```python
# http.py — the only way to call SEC. Enforces <=10 req/s, the configured User-Agent,
# exponential backoff on 429/5xx; raises PermanentFetchError on other 4xx.
def sec_get(url: str) -> bytes

# objects.py — content-addressed, immutable storage under settings().data_dir / "objects"
def put_object(data: bytes, suffix: str = "") -> tuple[str, str]   # (sha256, relative path)
def read_object(path: str) -> bytes

# documents.py — deterministic HTML/text -> normalized plain text. Claim spans and chunk
# offsets index into exactly this string, so it must be stable for a given stored object.
def load_text(filing: Filing) -> str

# edgar.py / xbrl.py / market.py
def ingest_filings(s: Session, ticker: str, quarters: int = 6) -> int     # rows inserted
def ingest_xbrl_facts(s: Session, ticker: str) -> int
def ingest_market_data(s: Session, ticker: str, start: date, end: date) -> int

# flows.py — the five §10.1 Prefect flows, each idempotent on its natural key and writing
# ingest_failure rows instead of raising on permanent failure:
# ingest_edgar_filings, ingest_xbrl_facts, ingest_call_audio, ingest_fomc_eval_set,
# ingest_market_data.  CLI: python -m ledgerline.ingest backfill --ticker AAPL [--all]
```

## Typed facts — `ledgerline/store/xbrl_query.py`

```python
@dataclass(frozen=True)
class Fact:
    value: float; unit: str; fact_ids: tuple[int, ...]; concept: str
    fiscal_year: int; fiscal_period: str; period_end: date; derived: bool; source_url: str

def get_xbrl_fact(s: Session, cik: str, concept: str, fiscal_year: int,
                  fiscal_period: str) -> Fact | None
# concept is a CONCEPTS key. Quarterly periods return 3-month values, FY returns 12-month.
# Derived concepts (margins, FCF) compose component facts and carry all their fact_ids.
# Returns None when the company never filed a mapped tag for that period (R8): never a guess.

def get_series(s: Session, cik: str, concept: str, fiscal_period_kind: str = "Q") -> list[Fact]
# Quarterly history, oldest first, for the forecast agent.
```

## S2 Audio — `ledgerline/audio/`, plus `ingest/fomc.py`, `ingest/call_audio.py`

```python
def ingest_call_audio(s: Session, path_or_url: str, *, kind: str, company_id: int | None,
                      event_date: date | None, license_note: str) -> int   # call_audio.id
def ingest_fomc_eval_set(s: Session, dates: list[str]) -> list[int]
def process_call(s: Session, call_audio_id: int) -> int                  # segments written
# VAD (Silero, 250 ms min speech, 200 ms pad) -> faster-whisper int8 word timestamps ->
# role attribution -> prepared-remarks/Q&A split -> segment rows.
def segment_text(s: Session, call_audio_id: int) -> tuple[str, list[tuple[int, int, int]]]
# Concatenated transcript + (segment_id, char_start, char_end) map, for extraction.
```

## S3 Evidence store — `ledgerline/store/`

```python
def chunk_document(text: str, form_type: str) -> list[ChunkDraft]   # section-aware, ~800/100
def index_filing(s: Session, filing_id: int) -> int                 # chunks written + embedded
def search(s: Session, query: str, *, company_id: int, k: int = 10,
           filing_ids: list[int] | None = None, rerank: bool = True) -> list[Hit]
# Hit(chunk_id, filing_id, section_path, text, score, char_start, char_end)
# BM25-style leg: Postgres tsvector + ts_rank_cd; dense leg: pgvector cosine (hnsw);
# fused with reciprocal rank fusion (k=60); optional cross-encoder rerank.
```

## S4 Extraction — `ledgerline/extract/`

```python
def extract_claims(text: str, *, fiscal_year: int, fiscal_period: str,
                   speaker_roles: list[tuple[int, int, str]] | None = None) -> list[ClaimDraft]
def extract_for_filing(s: Session, filing_id: int) -> int    # claim rows written (8-K EX-99.1)
def extract_for_call(s: Session, call_audio_id: int) -> int
def link_across_quarters(s: Session, company_id: int) -> int # guidance -> later actual
EXTRACTOR_VERSION: str
```

## S5/S6 Verification graph — `ledgerline/graph/`

```python
def verify_claim(s: Session, claim: Claim, tracer: Tracer) -> VerifyResult
# VerifyResult(verdict: VerdictDraft, evidence: list[EvidenceItem], critic: CriticOutput | None,
#              refused: bool, prompt_version: str, model: str | None, cost_usd: float,
#              usages: list[tuple[str, Usage]])
# router (deterministic, no LLM) -> concurrent evidence agents -> refusal short-circuit if
# every agent is `insufficient` -> critic -> adjudicator. Built as a LangGraph StateGraph.
def route(claim: Claim) -> list[AgentName]      # pure function, unit-tested without a model
```

## S7 Report and API — `ledgerline/report/`, `ledgerline/api/`, `ledgerline/pipeline.py`

```python
def severity(verdict_label, confidence, deviation, hedging_score, abnormal_return,
             weights: SeverityWeights) -> float   # §10.7, config-driven weighted sum
def run_report(s: Session, report_id: UUID) -> None    # the whole flow for one company-quarter
```

API (§15): `POST /v1/reports` -> 202 `{job_id}`; `GET /v1/reports/{id}`; `GET /v1/reports`
(cursor pagination); `GET /v1/claims/{id}` (claim + evidence + verdict + trace);
`GET /v1/companies/{ticker}/claims`; `GET /v1/evals/latest` (public).

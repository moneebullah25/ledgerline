# Ledgerline PRD (reconstructed)

> The original PRD (`LEDGERLINE_PRD.md`, v1.0, 2026-08-09) is missing. This is a reconstruction
> from `DevBooks/LEDGERLINE_ROADMAP.md`, which cites the PRD by section and requirement ID.
> Anything stated in the roadmap is kept as stated. Anything filled in from context is
> marked **(inferred)**. Where the roadmap gives only an ID and no content, this says so
> rather than inventing one. Section numbers follow the PRD's own numbering.

## §3 Goals and non-goals

Goal: verify what management says in earnings releases and on earnings calls against what
the company filed with the SEC, and produce a severity-ranked report where every number is
cited. The things the product must never give up are the eval harness, citation integrity
and the refusal path (§19).

Non-goals **(inferred)**: investment advice; saying anyone intended to mislead (see the
vocabulary rule in §16); real-time/intraday coverage; banks and other filers whose revenue
concepts do not map to the standard tags (`concepts.py` leaves them out on purpose).

## §4 Primary flow

An analyst picks a ticker and a quarter and gets back a ranked, cited report. In the build,
that is `python -m ledgerline.pipeline --ticker AAPL --fy 2025 --fp Q3`, or
`POST /v1/reports` followed by `GET /v1/reports/{job_id}`.

## §5 Scope tiers

| Tier | Weeks | Content **(inferred beyond the week ranges)** |
|---|---|---|
| v0.1 | 1-2 | ingest, schema, retrieval |
| v0.5 | 3-6 | audio, extraction, verification graph, first end-to-end verdicts |
| v1.0 | 7-10 | eval harness L1-L4, eval-gated CI, report/API, deployment, write-up |

## §7 Architecture

§7.2: four decisions, each recorded with a rejected alternative (see `docs/adr/`):
1. One Postgres + pgvector for everything (ADR-0001).
2. LangGraph state graph, not chains (ADR-0002).
3. Agents specialized by evidence source, not persona (ADR-0003).
4. Typed fact queries, not text-to-SQL (ADR-0004).

§7.3: repo layout `ledgerline/{ingest,audio,store,extract,graph,report,api,evals}`.

## §8 External sources

- SEC EDGAR / companyfacts: descriptive `User-Agent` with contact address required (an
  anonymous one gets 403); at most 10 requests/s; back off on 429/5xx, never treat 429 as
  permanent.
- Schema drift (a vanished XBRL tag, a changed IR page layout) is recorded as a diagnostic
  row, not a crash.
- Earnings-call audio: licensed for processing, not for redistribution (§16).
- FOMC press-conference audio + official transcripts: the public-domain ASR benchmark.

## §9 Data model

`company`, `filing`, `xbrl_fact`, `chunk` (tsvector + HNSW vector index), `call_audio`,
`segment`, `claim`, `report`, `evidence`, `verdict`, `span`, `market_price`,
`ingest_failure`, `eval_run`. Provenance columns (`source_url`, `fetched_at`, `sha256`,
`license_note`) are NOT NULL on ingested rows; idempotency keys are unique constraints.
Implemented in `ledgerline/models.py` as three Alembic migrations.

## §10 Subsystems

| § | Subsystem | Key requirements from the roadmap |
|---|---|---|
| 10.1 | Ingest (S1) | Five Prefect flows: `ingest_edgar_filings`, `ingest_xbrl_facts`, `ingest_call_audio`, `ingest_fomc_eval_set`, `ingest_market_data`; each idempotent on its natural key; permanent failures written to `ingest_failure`; independently retryable (NFR-4) |
| 10.2 | Audio (S2) | Silero VAD (250 ms min speech, 200 ms pad) -> faster-whisper int8 with word timestamps -> speaker roles (intro regex -> embedding clusters -> majority vote) -> prepared-remarks / Q&A split on the operator's question-instruction phrase |
| 10.3 | Evidence store (S3) | Section-aware chunks (~800 tokens, 100 overlap, never crossing a section); BM25-style + dense legs fused with RRF; cross-encoder rerank |
| 10.4 | Extraction (S4) | NER (GLiNER zero-shot; fine-tune DeBERTa-v3 once ~500 labeled spans exist) -> claim typing, dropping the majority `non_claim` class early -> schema-constrained normalization (§22.1); hedging score; guidance linked to later actuals |
| 10.5 | Verification graph (S5) | Deterministic router (no LLM, unit-tested) -> four concurrent evidence agents -> refusal short-circuit when all are insufficient -> adversarial critic -> adjudicator |
| 10.6 | Forecast agent (S6) | Baseline ensemble (seasonal naive / ETS / AutoARIMA) + Chronos-Bolt; if MASE is not < 1.0 vs. seasonal naive, its stance is not used |
| 10.7 | Report (S7) | Severity ranking as a config-driven weighted sum (label x confidence, deviation magnitude, lack of hedging, market reaction), tunable without code changes |
| 10.8 | Model tiering and cost | Cheap tier for high-volume steps, mid tier for agents and critic, top tier for the adjudicator; prompt caching on stable system prompts, and a zero cache-read rate in traces is a bug |

## §11 Evaluation (four layers)

| Layer | What | Targets |
|---|---|---|
| L1 components | ASR WER (FOMC), VAD, retrieval, NER, speaker roles, forecast | WER <= 12%; VAD removes >= 30% duration at >= 99% speech recall; Recall@10 >= 0.80 with reranker lift reported; NER span F1 >= 0.75; role accuracy >= 0.85 or documented escalation; MASE < 1.0 |
| L2 claim level | Frozen golden set of 200 labeled claims (floor 120, R3); precision/recall/F1 per label; three ablations | Contradiction precision >= 0.70 (headline metric) |
| L3 judge calibration | LLM judge vs. human labels | Cohen's kappa >= 0.60; if lower, report it and fall back to L2 for gating (R4) |
| L4 temporal backtest | Flagged discrepancies vs. later outcomes over >= 40 (company, quarter) pairs | Precision >= 0.55 |

Golden-set protocol: blind labeling with a second pass; the set is frozen, and prompts are
never tuned against it.

## §12 LLMOps

- Trace every graph node (deterministic ones included), with tokens, cost and cache reads.
  The PRD names Langfuse; the build writes spans to Postgres (`span` table).
- Eval-gated CI: PRs touching `graph/`, `extract/` or `store/` run L2 on the golden set;
  merge is blocked if the headline metric drops > 2% vs. main. A per-claim verdict diff is
  posted for triage.
- Weekly drift monitoring: claim-type distribution, hedging score, verdict distribution,
  confidence.
- Prompt changes show up as a version diff and a traceable cost delta.

## §13 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Full report latency <= 20 min per company-quarter |
| NFR-2 | API p95 <= 300 ms on cached report retrieval |
| NFR-3 | Cost per report <= $3.00 |
| NFR-4 | Per-source ingest retries are independent; one failing flow does not block others |
| NFR-5 | Secrets only via environment / secret manager; fail closed when missing |
| NFR-6 | One-command reproducible local environment (`docker compose up` + `make backfill`) |
| NFR-7 | No PII beyond already-public names |
| NFR-8 | Every user-visible number traces to a fact ID or filing span |

## §14 Cost model

About $2.70 per report against the $3.00 ceiling (NFR-3), plus a GPU budget line for
on-demand ASR/embeddings (Modal, see Q3). The per-line breakdown is not recoverable from the
roadmap. Measured cost is `report.cost_usd`, the sum of per-call costs from real token counts.

## §15 API

`POST /v1/reports` -> 202 `{job_id}` (async job); `GET /v1/reports/{id}`; `GET /v1/reports`;
`GET /v1/claims/{id}` (claim + evidence + verdict + trace); `GET /v1/companies/{ticker}/claims`;
`GET /v1/evals/latest`, public, because the eval numbers are the point and must not sit
behind an internal script.

## §16 Security, legal, compliance

- EDGAR `User-Agent` compliance and rate limit (§8).
- Call audio is not redistributed.
- Prompt injection: filing and transcript text goes to the model only as a delimited data
  block in the user turn, never in the system prompt or an instruction position.
- Secrets never in code, images or prompts (prompts are persisted in traces).
- Defamation-safe vocabulary: output says `contradicted`, never "misleading", "false" or
  "fraudulent".

## §17 Risks

| ID | Risk | Mitigation |
|---|---|---|
| R3 | Golden-set labeling runs late or short | 200-claim target with a 120 floor; labeling starts in week 5 |
| R4 | Judge-human agreement too low to trust | Report kappa as measured; gate falls back to L2 |
| R8 | XBRL tag inconsistency across companies | Tag-alias map per canonical concept; an unmapped tag resolves to `unverifiable`, never a nearby number |
| R1, R2, R5-R7 | Content not recoverable from the roadmap | - |

## §18 Milestones (ten weeks)

| Week | Exit criterion (as quoted or paraphrased in the roadmap) |
|---|---|
| 1 | Docker up; 15 companies x 6 quarters of filings and facts; ingest re-run produces zero duplicates |
| 2 | Retrieval Recall@10 on hand-built pairs; reranker lift reported |
| 3 | FOMC WER and VAD duration reduction measured |
| 4 | Not recoverable **(inferred: audio roles/split and forecast baselines)** |
| 5 | >= 30 structured, schema-valid claims per call; NER F1 reported; golden-set labeling starts (the schedule's load-bearing task) |
| 6 | Verdicts end to end for 3 companies; refusal path exercised; every numeric assertion carries a fact ID |
| 7-8 | Eval harness: 200 labeled claims, L2 with ablations, kappa, backtest |
| 9 | A PR that regresses L2 > 2% is blocked by CI, demonstrated |
| 10 | One-command local reproduction; public write-up |

## §19 Descope order

Cut first: the P1/P2 requirements FR-5, FR-23, FR-33, FR-34, FR-52, FR-53. **Never cut: the
eval harness, citation integrity, the refusal path.**

## §20 Success metrics (to be reported as measured values, never as targets)

Claims per call >= 30; citation integrity 100%; contradiction precision (L2) >= 0.70;
backtest precision (L4) >= 0.55; judge-human kappa >= 0.60 (or honestly lower, with L2
fallback); FOMC WER <= 12%; cost per report <= $3.00; report latency <= 20 min. Portfolio
artifacts: public repo with one-command reproduction, write-up leading with eval tables and
ablation deltas, live public `/v1/evals/latest`, four ADRs each with a rejected alternative.

## §21 Open questions

Q3: self-host ASR and embeddings on on-demand GPU (Modal/Fly), chosen because cost per
report, not throughput, is the metric. Other questions are not recoverable.

## §22 Contracts

§22.1 claim (extraction output), §22.2 evidence-agent output (stance + citations, never a
verdict), §22.4 adjudicator verdict (label, confidence, rationale, citations; FilingRAG
citations must quote the span). §22.3 is not named in the roadmap **(inferred: the critic
output)**. Implemented in `ledgerline/schemas.py`.

## Functional requirements

P0 unless marked. Descriptions are from the roadmap's requirement map (§E.4).

| ID | Requirement |
|---|---|
| FR-1 | Idempotent EDGAR filing fetch keyed on accession number, rate-limit safe |
| FR-2 | XBRL companyfacts parsed into typed `xbrl_fact` rows with correct units and periods |
| FR-3 | Call audio stored content-addressed (sha256), no duplicate rows on re-fetch |
| FR-4 | FOMC audio + transcript eval set for ASR WER |
| FR-5 | P1/P2; content not recoverable |
| FR-6 | Provenance (`source_url`, `fetched_at`, `sha256`, `license_note`) on every ingested row |
| FR-7 | Idempotency keys, exponential backoff, `ingest_failure` rows on permanent failure |
| FR-8 | Referenced by ID only; content not recoverable |
| FR-10 | VAD before ASR (>= 30% duration removed) |
| FR-11 | ASR with word-level timestamps |
| FR-12 | Speaker role attribution |
| FR-13 | Prepared-remarks / Q&A boundary detection |
| FR-14 | Every transcript token resolves to (segment_id, t_start, t_end) |
| FR-20 | Section-aware chunking + embeddings in pgvector |
| FR-21 | Hybrid retrieval: BM25 + dense, RRF, cross-encoder rerank |
| FR-22 | Typed fact queries instead of model-generated numbers |
| FR-23 | P1/P2; content not recoverable |
| FR-30 | Financial-entity NER |
| FR-31 | Zero-shot claim typing; drop `non_claim` early |
| FR-32 | Schema-constrained claim normalization (§22.1) |
| FR-33, FR-34 | P1/P2; content not recoverable |
| FR-40 | Deterministic, unit-testable router |
| FR-41 | Four concurrent evidence agents |
| FR-42 | Adversarial critic: stance + argument, never a verdict |
| FR-43 | Schema-constrained adjudication with rationale and citations |
| FR-44 | Refusal path: no evidence -> `unverifiable` without an adjudicator call |
| FR-45 | Citation integrity as a hard validation gate |
| FR-50 | Severity-ranked report per company per quarter |
| FR-51 | REST API for reports and claim drill-down |
| FR-52, FR-53 | P1/P2; content not recoverable |

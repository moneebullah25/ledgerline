# Tester panel findings (2026-09-23) and fix ownership

Panel: testing-reality-checker, security-appsec-engineer, engineering-code-reviewer,
testing-api-tester, testing-test-results-analyzer. All reviewed statically (read-only tools).
Every item below is to be fixed. Group letters = fix owner. Mark each `[x]` when done.

## A. Verification graph + citation integrity (`ledgerline/graph/**`, `ledgerline/schemas.py`, `ledgerline/llm.py`)
- [x] A1 FR-45 bypass: `_adjudicate_llm` trusts model-supplied citation `value`/`quote`. Rebuild every returned citation from the evidence Citation with the same (kind, ref_id); drop unknown ones. (adjudicator.py:131-139)
- [x] A2 `uncited_numbers` exemptions too wide: ints <32 and 1990-2100 are exempt even next to `$`, `%`, scale words ("$25 billion", "$2,050 million", "grew 12%"). Exempt only bare ordinals/years not adjacent to $/%/scale. Allow abs() renderings for negative cited values (net loss). Add negative tests. (schemas.py)
- [x] A3 Label/stance consistency: online adjudicator label must agree with the deciding agent's stance from `decide()` (XBRL decides numeric claims); otherwise retry once, then fall back to the offline rule.
- [x] A4 Forecast-only verdicts: when the forecast agent is the only deciding evidence, cap the label at `unverifiable` (user decision). Forecast evidence stays visible.
- [x] A5 Prompt-injection delimiter breakout: `<document label=..>{text}</document>` is unescaped; filer text can contain `</document>`. Use a per-call random nonce tag and neutralise the tag inside text; JSON evidence blocks too. (llm.py:74-76)
- [x] A6 Retry instruction puts untrusted `problems` (model ref_ids, numbers) in instruction position; move detail into a data block, keep only counts/kinds in the instruction. (adjudicator.py:142)
- [x] A7 LLM-generated summaries (FilingRAG `j.summary`, online critic `argument`) are only vocabulary-checked; also run `uncited_numbers`. (filing_rag.py:124-131, critic.py:88-91)
- [x] A8 Derived facts: cite each component fact with its OWN value and unit; computed ratio/diff goes in payload (and as a `kind="xbrl_fact"`-free computed citation or payload value the audit accepts). Don't set derived=True on ratios merely for being computed (xbrl_query.py:54 — coordinate: small edit OK). (xbrl_table.py:103-104)
- [x] A9 Computed change values (yoy actual %, tolerance) are not citable, so a correct online rationale "grew 5.4%" fails the audit. Add a citation for the computed change.
- [x] A10 Comparison claims skip the unit check: require claim.unit in ("percent", concept.unit) else insufficient. (xbrl_table.py:118,142-144)
- [x] A11 Tolerances: percent-unit level claims should use PP_TOL not REL_TOL*|actual|; USD change claims should scale with REL_TOL*(|cur|+|base|); `printed_half_ulp` must prefer tokens followed by %/points for percent claims. (xbrl_table.py:45-49,129,144)
- [x] A12 Loss sign: "net loss of $X" / "operating loss" — verify claim sign vs negative NetIncomeLoss; compare magnitudes consistently so a correct loss claim is not contradicted. (with extraction group E if the fix belongs there)
- [x] A13 FilingRAG holds `db_lock` during embedding + rerank inference, serialising agents; hold it only for DB reads. (filing_rag.py:73-76)
- [x] A14 `verify_claim` span opened twice (pipeline.py:151 and build.py:161) — keep one (the build.py one); group B removes the pipeline one.
- [x] A15 Tests with a stubbed `llm.structured`: adjudicator (bad then good; both bad -> IntegrityError; fabricated value rejected; label/stance mismatch), judge_llm (non-verbatim quote rejected, banned word), online critic. Refusal graph tests: xbrl insufficient + filing_rag supports -> no refusal and adjudicator called; forecast-only guidance path; qualitative (no xbrl) path through the graph; refusal text included in the vocabulary test; router cases value=0.0, guidance with one target field, negative change claim vs `mentions_value` abs().

## B. Platform (`pipeline.py`, `worker.py`, `api/**`, `report/**`, `models.py` + ALL new migrations, compose, Dockerfile, Makefile, CI)
- [x] B1 `_deviation` reads `relative_error`; agent writes `deviation`. Read `deviation` (abs), delete the fallback; fix the test that seeds the wrong key. (pipeline.py:68-74, tests/test_report_generate.py:74)
- [x] B2 One claim's IntegrityError fails the whole report: catch per claim, persist as refused/unverifiable with the reason, continue. (pipeline.py:152-159)
- [x] B3 Report-time audit includes the verdict's own citation copy (`shown = cites + sources`): audit against resolved `sources` only. (report/generate.py:139)
- [x] B4 Evidence latency always 0: span names are `agent.<name>`; look up `latency.get(f"agent.{agent}", ...)`. Remove the duplicate `verify_claim` span in pipeline. (pipeline.py:89,100,151)
- [x] B5 `report.cost_usd` lost on crash: compute from sum(Verdict.cost_usd) (+ evidence) at finish and on failure.
- [x] B6 Worker stale sweep can double-run: heartbeat (`config.heartbeat_at`) refreshed after each claim commit; sweep keys off heartbeat.
- [x] B7 `/v1/evals/latest` shows one row per layer, hiding 3 of 4 L1 evals: add `eval_run.name` (migration 0004) and return latest per (layer, name); update writers only via a default so other modules keep working (name defaults to layer) — then set names in evals/*.py writers (retrieval, ner, wer, forecast: one-line edits OK).
- [x] B8 Auth: require an API key header (`LEDGERLINE_API_KEY`, fail closed if unset when POST is used) on POST /v1/reports; dedupe: return the existing queued/running/done report for the same (company, fy, fp) instead of inserting; cap queued reports.
- [x] B9 Cursor 500: `id_.isdigit()` accepts `²` and >4300-digit strings; use isascii()+isdigit() with a length bound / try-except -> 422. Empty cursor -> 422. Consistent 422 error shape (list form like FastAPI's). 202 gets a `Location` header. (api/main.py:241-243)
- [x] B10 Tracebacks exposed via `report.error` and span attrs: store an error class + short message for the API; keep traceback server-side (log). (pipeline.py:159, tracing.py:32)
- [x] B11 `market_price` provenance (FR-6): add sha256 + license_note (migration 0004), filled by ingest/market.py (one-line edit OK) with a Yahoo license note.
- [x] B12 `ingest_failure` unique (flow, key, error_class) in migration 0004 named `uq_ingest_failure` (group C switches record_failure to on_conflict).
- [x] B13 Compose: bind db and api ports to 127.0.0.1; api container must not get ANTHROPIC_API_KEY (worker only); `.env.example` POSTGRES_PASSWORD empty so `:?` fails closed.
- [x] B14 CI: eval-gate — scope ANTHROPIC_API_KEY to the gate step, `persist-credentials: false`, pin actions to commit SHAs, pass base_ref via env; ci.yml top-level `permissions: contents: read`, persist-credentials false; ruff must pass on the whole repo (fix lint across repo at the end, coordinate last).
- [x] B15 Dockerfile worker image: install ffmpeg + poppler-utils (pdftotext) when EXTRAS includes ml.
- [x] B16 FY reports always fail (releases labelled Q4): map --fp FY to the Q4 release, or reject FY with a clear 422/CLI error. Makefile backfill includes audio (FOMC eval set) and `make evals` runs all evals. Call-audio flow must pass fiscal_year/period (ingest/flows.py).
- [x] B17 Offline mode is silent: report view shows `mode: offline (no LLM credentials)` when verdict.model is NULL.
- [x] B18 Perf test `test_nfr2_report_p95`: mark `perf`, excluded by default, still runnable.
- [x] B19 NFR-3/NFR-1: record and surface cost and latency per report in the report view; warn when over $3.00 / 20 min.

## C. Ingest + test infrastructure (`ledgerline/ingest/**` except market.py one-liner, `tests/conftest.py`, `config.py`, `tests/test_ingest_*`)
- [x] C1 conftest: per-session test database name (pid/uuid suffix), drop it at session end; force `LEDGERLINE_LLM_ENABLED=false` by assignment and delete ANTHROPIC_API_KEY/AUTH_TOKEN (autouse); resolve .env/alembic.ini relative to the repo root; skip cleanly if Postgres is unreachable or CREATEDB is denied; `pytest.importorskip` in ml tests.
- [x] C2 FR-7 idempotency test bypasses the DB constraint (`known` pre-filter): add a test calling `_store` twice. (ingest/edgar.py:45,53,152)
- [x] C3 `record_failure` check-then-insert races: use on_conflict against `uq_ingest_failure` (from B12). Exhausted transient retries (`TransientFetchError`, `httpx.TransportError`) must become `http_transient_exhausted` rows instead of aborting the ticker. Test both.
- [x] C4 XBRL re-run with a CHANGED payload (amendment changes a YTD base): derived quarters must not silently keep a stale value; test it.
- [x] C5 429 retry honours `Retry-After`. SEC throttle is per-process: document (ponytail comment) — no cross-process lock.
- [x] C6 `sec_user_agent` must contain a contact (email address pattern); fail closed otherwise. (config.py)
- [x] C7 `read_object` must resolve under data_dir (reject traversal). Download size caps on http.get / call_audio / fomc.
- [x] C8 fomc.py: `urljoin(master, uri)` for relative HLS URIs, require https, ffmpeg `-protocol_whitelist https,tls,tcp,crypto,hls`. call_audio.py: https-only host allowlist + audio suffix allowlist (local paths allowed only under an explicit directory).
- [x] C9 html_to_text: also drop `visibility:hidden`, `font-size:0` styled elements (injection hiding). Keep load_text byte-stable for existing stored objects (existing chunk offsets must still slice exactly — verify).
- [x] C10 Market prices never re-based after splits/dividends: refresh the last N days on each run (upsert adj_close) or document with the ponytail comment already present — implement the upsert.

## D. Evaluation harness L2-L4 (`ledgerline/evals/{l2,l3,l4,golden}.py`, `ledgerline/evals/data/golden*.jsonl`, `evals/baseline.json`, eval-gate wiring with B)
- [x] D1 Frozen golden set, 200 claims (floor 120), stratified by company/type/metric with a fixed seed, labeled blind to system output with source evidence recorded per label; plus a clearly separated perturbation set (real claims with injected wrong values/periods) because real contradictions are rare. Second-pass review of the most consequential labels; report inter-pass agreement.
- [x] D2 L2: precision/recall/F1 per label, contradiction precision as headline, three ablations (e.g. without reranker, without critic, xbrl-only), `python -m ledgerline.evals.l2`, `run()` returns the dict gate.py expects; writes eval_run (layer L2).
- [x] D3 L3: judge vs human agreement (Cohen's kappa) with the judge implemented behind llm.available(); offline, state plainly that no LLM judge ran and report what WAS measured (e.g. second-pass annotator agreement) without dressing it up as judge kappa.
- [x] D4 L4: temporal backtest over >=40 (company, quarter) pairs: guidance claims flagged by the forecast agent vs the later actual (linked claims / XBRL) -> backtest precision on flagged; writes eval_run (layer L4).
- [x] D5 CI gate runs against seeded data (fixture DB dump or golden-set fixtures), records a baseline on main, blocks >2% regression; document absolute-vs-relative choice vs PRD §12 wording.

## E. Extraction quality (`ledgerline/extract/**`, `tests/test_extract_*`, `ledgerline/evals/ner.py`)
- [x] E1 Non-earnings 8-Ks stored as releases (NKE CFO-transition, filing 176): extraction must not count them; flag/skip releases without results content (coordinate with ingest: an `is_earnings_release` check in extract is enough).
- [x] E2 Six false guidance claims (AMZN "$1B Prime savings", CSCO "AI revenue $4B", ADBE headline lines x2, HD fragments x2): tighten guidance typing.
- [x] E3 Full-year figures in Q4 releases stamped as the quarter (META $200.97B): detect "full year"/"fiscal year"/twelve-months context and set target period FY.
- [x] E4 Segment/partial actuals linked as company totals ("Revenues for Converse", INTC/AMZN segment figures): tighten segment qualifiers.
- [x] E5 `non_claim` bucket precision 0.611 (real claims discarded): reduce false discards (3 comparisons, 2 guidance, 2 reported results in the gold set).
- [x] E6 Guidance recall: parse guidance tables and cue-less forward-looking bullets where the scale and target period are explicit.
- [x] E7 Re-run extraction over all releases, re-link, and report the before/after table (claims per release, guidance precision on a hand check, NER/typing eval). Keep gold sets frozen (do not tune against them; report results on them honestly).

# Round 2 — panel re-check (2026-09-24): testing-reality-checker, engineering-code-reviewer, security-appsec-engineer

## F. Platform (`pipeline.py`, `worker.py`, `api/**`, `report/**`, `.github/**`)
- [x] F1 Report source values: upserts can now change cited rows; `_source` must keep the verdict's cited value when the row differs and add a "since revised: <new value>" note, and the report-time audit must accept the cited value.
- [x] F2 POST /v1/reports: add `rerun: bool = false` (still key-gated) so a done report can be recomputed; default stays deduped.
- [x] F3 Heartbeat per extracted filing/call inside `_ensure_claims`; refresh `report.config` from the DB before writing the heartbeat (no stale-dict clobber of `attempts`).
- [x] F4 eval-gate path filter: add `ledgerline/schemas.py`, `ledgerline/concepts.py`, `ledgerline/llm.py`, `ledgerline/evals/**`, `ledgerline/ingest/**`.
- [x] F5 B2 widened: any per-claim exception (model refusal/ValueError, API/overload errors, adjudicator wiring error) becomes a refused `unverifiable` verdict whose rationale names the error class; spend recorded from spans and `err.usage`. Only DB/infrastructure errors fail the report.
- [x] F6 Mode: record `report.config["mode"]` ("online"/"offline") from `llm.available()` at run time and show that, not an inference from verdict.model. Integrity/error verdicts render as "(no verdict: <reason>)", not "(refused: no evidence)".
- [x] F7 Report layout: group findings by label (contradicted, then unverifiable, then supported), severity-ranked within each group, so refusals never sit above findings.

## G. Ingest (`ingest/**`, `store/__main__.py`)
- [x] G1 Upserts compare the full stored tuple with IS DISTINCT FROM (xbrl: value, derived, fiscal_year, fiscal_period, duration_days, form, filed; market: close, adj_close, volume, sha256), so parser fixes and pre-0004 `sha256='unrecorded'` rows are corrected. Deviation: market compares close/adj_close within 1e-6 relative (Yahoo float32 wobble) plus volume, and rewrites sha256 only when stored `unrecorded` (the response hash differs every call). Backfill: facts 0, prices 12012 corrected.
- [x] G2 `python -m ledgerline.store verify-offsets`: re-slices every stored chunk (and claim span) against current `load_text`; exits non-zero on any mismatch. Run it and record the result. Result 2026-09-23 on main DB: 12052 offsets verified, 0 mismatches.

## H. Graph (`graph/**`, `schemas.py`) — after the L2 agent finishes
- [x] H1 Adjudicator label-mismatch fallback: use the offline critic (or none), attach usage to any IntegrityError, and return usage with model=None so the verdict is not labelled online.
- [x] H2 A3 gap: label mismatch plus another problem after the retry also falls back to the offline rule (not raise).
- [x] H3 Loss sign: `_LOSS` matches "loss from operations"; the `_INCOME` veto applies only to the mention attached to the value; change claims on a negative base ("net loss widened 20%") compare magnitudes.
- [x] H4 `printed_half_ulp` handles basis points ("120 basis points" = 1.20 pp).
- [x] H5 `uncited_numbers`: currency prefixes € £ USD, hyphenated scales ("25-billion"), any whitespace before the unit.

## I. Extraction + evals (`extract/**`, `concepts.py` synonyms, `evals/l2.py`, `evals/l4.py`)
- [x] I1 Segment claims keep their qualifier in the claim text ("Data Center revenue was $89.0 billion"), and overlapping-span duplicates are merged for metric=None claims too.
- [x] I2 Loss synonyms: "net loss", "operating loss", "loss from operations", "net loss per share" extract as the signed concept with a negative value.
- [x] I3 L4 and `l2 --current` filter on the current EXTRACTOR_VERSION; `--current` tie-break prefers the newest claim; L2 docstring/ERROR mapping matches the pipeline (refused unverifiable).

# Round 3 — whole-app panel on the online system (2026-09-25): reality-checker, code-reviewer, security, api-tester, test-results-analyzer, performance-benchmarker

## J. LLM backend + extraction (`llm.py`, `tracing.py`, `extract/**`, tests)
- [x] J1 CLI isolation: add `--no-session-persistence --setting-sources "" --disable-slash-commands` (verified with the installed CLI 2.1.281: a call with these left no ~/.claude/projects folder) and run the child with an allowlisted env (HOME, PATH, USER, TMPDIR, LANG only — no DB URL, API keys); test that argv carries the flags and never the prompt, and the cwd is a fresh temp dir.
- [x] J2 Infrastructure errors: CLI auth/logged-out errors, subprocess.TimeoutExpired (after retries), and SDK RateLimitError/APIConnectionError/APITimeoutError raise an infra error (`llm.RateLimited` or a sibling `LLMUnavailable`), never a per-claim ValueError. Transient detection must not match bare "429"/"529" inside unrelated text (match HTTP status / known phrases). `out['result']` may be null.
- [x] J3 No model output or raw stderr in exception messages that reach the public API: wrap pydantic ValidationError in a fixed-message ValueError; stderr to the log.
- [x] J4 refine: comparison claims may only be re-mapped to a concept with the same unit (blocker: "gross margin expanded 120 bps" -> gross_profit growth). Test with that input.
- [x] J5 refine runs after `_finish` dedupe and role filtering (no calls for dropped duplicates/analyst turns; no divergence between restatements).
- [x] J6 refine table context stops scanning upward at a value row once a header is collected (no header rows from a previous table).
- [x] J7 refine cache: unique temp file (mkstemp) + os.replace; a corrupt/unparseable cache loads as empty with a warning; key includes model id and Normalized schema; a cached entry that no longer validates is re-asked.
- [x] J8 refine returns its llm.Usage total so the caller can count it (the pipeline adds it to report cost, group K).
- [x] J9 EXTRACTOR_VERSION by mode: "s4-llm-1" only when the LLM pass ran; offline extraction writes the rules version, so claims are re-extracted when credentials appear.
- [x] J10 Heartbeat hook: `extract_for_filing(..., on_progress=None)` calls it per completed batch of refine calls, so the pipeline can refresh the report heartbeat during long LLM extraction.

## K. Platform (`pipeline.py`, `worker.py`, `api/**`, `report/**`, `db.py`, `models.py` + migration 0005, tests)
- [x] K1 int4 bounds: `claim_id` path param and claim cursor ids bounded to 1..2**31-1 -> 422, never a 500 (test 2147483648 and "9"*36).
- [x] K2 GET /v1/claims/{id}: verdict/evidence from the latest `done` report only (or document otherwise in CONTRACTS.md and the docstring).
- [x] K3 /healthz returns JSON 503 when the DB is down.
- [x] K4 ReportRequest strict (ConfigDict(strict=True)), reject quarters after the current date; document the soft queue cap (ponytail comment) or make it hard with a global advisory lock.
- [x] K5 Migration 0005: index verdict(claim_id, id); Evidence.cache_write_tokens column (persist it); size the DB pool to the worker concurrency.
- [x] K6 Parallel claim verification in run_report: ThreadPoolExecutor (default 4, config), one Session per worker for verify_claim, persist/heartbeat/commit on the main thread as futures complete; a shared rate-limit gate is not required if RateLimited fails the report.
- [x] K7 Budget guard (NFR-3): when the running cost exceeds budget x done/total, later claims use the offline critic and adjudicator (pass a flag through verify_claim); record in report.config that the guard fired.
- [x] K8 Cost accounting: refine's Haiku usage (J8) is added to report.cost_usd; include cache writes.
- [x] K9 Infra errors (J2's LLMUnavailable/RateLimited, auth) fail the report (re-queueable) instead of per-claim no-verdicts.
- [x] K10 Tests: a real two-connection concurrency test for POST dedupe; RateLimited through verify_claim fails the report (pipeline) — mirror the OperationalError test; API gaps (limit bounds on company claims, claim without verdict, failed report error field, 401 WWW-Authenticate).

## L. Verification graph (`graph/**`, `schemas.py`, tests)
- [x] L1 Critic gating: use `critique_offline` when `not adjudicator.needs_judgment(...)` (no Sonnet critic on unopposed filed-fact verdicts); the offline rationale never quotes an online critic whose verdict is model=None.
- [x] L2 Implausible-gap abstentions get their own rationale ("filed fact differs by X%, likely a period or scope misreading") and are surfaced in the report as such — never the generic "no truth-bearing agent found evidence". Restrict the guard to USD and USD/share; percent concepts are measured in points (a margin 13 points off is a contradiction to rule on).
- [x] L3 Scoped claims: a claim with a value but no metric (extraction judged it not a company total) may not be contradicted by FilingRAG against a company-level reading; the judge must see the scope (claim_context) and a contradiction requires a passage about the same scope and period. Enforce the period rule in code: a `contradicts` whose quoted passage's filing period differs from the claim's period becomes insufficient.
- [x] L4 claim_context: return {} unless text[span_start:span_end] == claim.text; parse load_text outside db_lock with a per-filing cache.
- [x] L5 forecast `_chronos` model load under a lock; `deviation` when actual == 0 must not feed severity as an absolute amount.
- [x] L6 Tests: adjudicator rejects a non-unverifiable verdict whose citations are all unknown; needs_judgment's filing_rag branch with gating live; guard boundaries (49%/51%, over-claim, guidance/comparison excluded, range nearest bound); filing_rag negatives (different number in quote, passage=-1, empty quote); online `run()` path incl. claim_context (db test).

## M. Evaluation (`evals/**`, held-out data, tests)
- [ ] M1 Held-out golden set: >= 100 claims sampled (fixed new seed, stratified by company/type) from current s4-llm-1 claims NOT in golden.jsonl, labelled blind by the annotator BEFORE the system runs on them (same protocol and evidence recording as golden.jsonl), frozen in `evals/data/heldout.jsonl`; `python -m ledgerline.evals.l2 --set heldout`; report metrics with 95% Clopper-Pearson intervals (contradiction precision/recall, accuracy) in both L2 variants.
- [ ] M2 L3 robustness: per-claim ValueError recorded (not aborting), evidence missing for unmatched/errored ids skipped and counted, workers validated (>=1).
- [ ] M3 l2 workers capped below the DB pool; document in the eval-gate workflow that the gate covers the offline system only and how to record an online baseline.

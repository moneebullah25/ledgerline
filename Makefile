# One-command local reproduction (NFR-6): make up && make backfill TICKER=AAPL && make report TICKER=AAPL FY=2025 FP=Q3
# Model-backed steps run in the worker image, which has the [ml] extra and the ./data volume.

RUN := docker compose run --rm worker
# FOMC press conferences for the L1 ASR/WER eval set (YYYYMMDD, space-separated).
FOMC_DATES ?= 20240918
unexport VIRTUAL_ENV  # uv must use this project's .venv

.PHONY: up down migrate test backfill report evals

up:
	docker compose up -d --build db api worker

down:
	docker compose down

migrate:
	uv run alembic upgrade head

test:
	uv run ruff check .
	uv run pytest -m "not ml and not network and not perf"

backfill:
	@test -n "$(TICKER)" || (echo "usage: make backfill TICKER=AAPL" && exit 1)
	$(RUN) python -m ledgerline.ingest backfill --ticker $(TICKER)
	$(RUN) python -m ledgerline.store index --all
	$(RUN) python -m ledgerline.extract --ticker $(TICKER)
	$(RUN) python -c "import sys; from ledgerline.ingest.flows import guarded; \
		from ledgerline.ingest.fomc import ingest_fomc_eval_set as f; d = sys.argv[1:]; \
		print('fomc:', guarded('ingest_fomc_eval_set', ','.join(d), f, d))" $(FOMC_DATES)

report:
	@test -n "$(TICKER)" -a -n "$(FY)" -a -n "$(FP)" || (echo "usage: make report TICKER=AAPL FY=2025 FP=Q3" && exit 1)
	$(RUN) python -m ledgerline.pipeline --ticker $(TICKER) --fy $(FY) --fp $(FP)

# Every layer: L1 (retrieval, NER/typing, forecast, ASR WER), L2-L4, then the L2 regression gate.
evals:
	$(RUN) python -m ledgerline.evals.retrieval
	$(RUN) python -m ledgerline.evals.ner
	$(RUN) python -m ledgerline.evals.forecast
	$(RUN) python -m ledgerline.evals.wer --date $(firstword $(FOMC_DATES))
	$(RUN) python -m ledgerline.evals.l2
	$(RUN) python -m ledgerline.evals.l3
	$(RUN) python -m ledgerline.evals.l4
	$(RUN) python -m ledgerline.evals.gate

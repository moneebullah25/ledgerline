# One-command local reproduction (NFR-6): make up && make backfill TICKER=AAPL && make report TICKER=AAPL FY=2025 FP=Q3
# Model-backed steps run in the worker image, which has the [ml] extra and the ./data volume.

RUN := docker compose run --rm worker
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
	uv run pytest -m "not ml and not network"

backfill:
	@test -n "$(TICKER)" || (echo "usage: make backfill TICKER=AAPL" && exit 1)
	$(RUN) python -m ledgerline.ingest backfill --ticker $(TICKER)
	$(RUN) python -m ledgerline.store index --all
	$(RUN) python -m ledgerline.extract --ticker $(TICKER)

report:
	@test -n "$(TICKER)" -a -n "$(FY)" -a -n "$(FP)" || (echo "usage: make report TICKER=AAPL FY=2025 FP=Q3" && exit 1)
	$(RUN) python -m ledgerline.pipeline --ticker $(TICKER) --fy $(FY) --fp $(FP)

evals:
	$(RUN) python -m ledgerline.evals.l2
	$(RUN) python -m ledgerline.evals.gate

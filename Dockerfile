# syntax=docker/dockerfile:1
# Build: dependencies resolved from uv.lock into /app/.venv. EXTRAS=ml adds the model stack
# (worker image); the API image leaves it out and stays small.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Dependencies first, so editing application code does not reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}
COPY alembic.ini ./
COPY migrations ./migrations
COPY ledgerline ./ledgerline
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS:+--extra $EXTRAS}

# Runtime: same Python base as the build stage so the venv's interpreter links resolve;
# no uv, no build cache, no secrets (they arrive as environment variables at run time).
FROM python:3.12-slim-bookworm
ARG EXTRAS=""
# The [ml] image decodes call/FOMC audio (ffmpeg) and FOMC transcript PDFs (pdftotext).
RUN if [ "$EXTRAS" = "ml" ]; then \
      apt-get update && apt-get install -y --no-install-recommends ffmpeg poppler-utils \
      && rm -rf /var/lib/apt/lists/*; \
    fi
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["uvicorn", "ledgerline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

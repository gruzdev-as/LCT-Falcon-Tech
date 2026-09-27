# Build context is the repo root: the backend imports common/ as a sibling package.
FROM python:3.12.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
# The lock names the training submodule as a path source; uv needs its metadata only.
COPY training/pyproject.toml ./training/
RUN uv sync --locked --no-dev --no-default-groups --group backend

COPY common/ ./common/
COPY backend/ ./backend/

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "backend.app.server:app", "--host", "0.0.0.0", "--port", "8000"]

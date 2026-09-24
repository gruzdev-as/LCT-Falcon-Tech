FROM python:3.12.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-default-groups --group init

COPY common/ ./common/
COPY init/ ./init/

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["python", "-m", "init.app.main"]

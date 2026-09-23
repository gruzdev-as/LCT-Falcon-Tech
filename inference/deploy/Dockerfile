# Build context is the repo root: the worker imports common/ as a sibling package.
FROM python:3.12.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

# The stub embedder needs nothing beyond `common`. A real model needs torch: build
# with `--build-arg DEPS_GROUP=inference`.
# TODO FIX
ARG DEPS_GROUP=common

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-default-groups --group ${DEPS_GROUP}

COPY common/ ./common/
COPY inference/ ./inference/

ENV PATH="/app/.venv/bin:${PATH}"

# Stream worker: no ports. Liveness is the heartbeat file, touched after every healthy iteration.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, sys, tempfile, time; p = os.getenv('INFERENCE_HEARTBEAT_PATH') or os.path.join(tempfile.gettempdir(), 'inference.alive'); sys.exit(time.time() - os.path.getmtime(p) > 60)"

CMD ["python", "-m", "inference.app.main"]

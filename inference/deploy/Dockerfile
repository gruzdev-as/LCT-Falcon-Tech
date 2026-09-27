# Build context is the repo root: the worker imports common/ as a sibling package and
# installs the model code from the training/ submodule.
FROM python:3.12.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

# The torch wheels on PyPI bundle CUDA and fall back to the CPU, so one image serves both.
# The offline flags and the cuBLAS workspace match training's own serving image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /app

# Always the full group: the embedder is picked at runtime by INFERENCE_EMBEDDER, and an
# image without torch would only fail there. The stub runs in this image too.
# The training package is a path dependency, so its sources must be here before the sync.
COPY pyproject.toml uv.lock ./
COPY training/ ./training/
RUN uv sync --locked --no-dev --no-default-groups --group inference

COPY common/ ./common/
COPY inference/ ./inference/

ENV PATH="/app/.venv/bin:${PATH}"

# Stream worker: no ports. Liveness is the heartbeat file, touched after every healthy iteration.
# The start period covers loading the model, which happens before the first heartbeat.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os, sys, tempfile, time; p = os.getenv('INFERENCE_HEARTBEAT_PATH') or os.path.join(tempfile.gettempdir(), 'inference.alive'); sys.exit(time.time() - os.path.getmtime(p) > 60)"

CMD ["python", "-m", "inference.app.main"]

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.routers import search
from backend.src.configs.constants import (
    API_PREFIX,
    APP_TITLE,
    CORS_ALLOWED_ORIGINS,
    DEFAULT_ERROR_STATUS,
    ERROR_STATUS_MAP,
)
from backend.src.configs.schemas import ErrorResponse
from common.src.exceptions import FalconError
from common.src.logging.logging import setup_logging
from common.src.redis.client import close_redis, init_redis_streams
from common.src.storage.client import ensure_bucket

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    await init_redis_streams()
    await ensure_bucket()
    logger.info("Backend ready")

    try:
        yield
    finally:
        await close_redis()


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(FalconError)
async def handle_domain_error(_: Request, exc: FalconError) -> JSONResponse:
    code = next(
        (status for typ, status in ERROR_STATUS_MAP.items() if isinstance(exc, typ)),
        DEFAULT_ERROR_STATUS,
    )
    if code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        logger.error("Request failed: %s (%s) %s", exc.message, exc.code, exc.details)

    body = ErrorResponse(error=exc.code, message=exc.message, details=exc.details)
    return JSONResponse(status_code=code, content=body.model_dump())


app.include_router(search.router, prefix=API_PREFIX)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe. Does not touch dependencies."""
    return {"status": "ok"}

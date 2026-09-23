import os
from http import HTTPStatus

from common.src.exceptions import FalconError, NotFoundError, StorageError, ValidationError

# CORS
CORS_ALLOWED_ORIGINS = ["http://localhost:5173"]

# Naming
APP_TITLE = os.getenv("APP_TITLE", "FastAPI server for LCT")

# Routing
API_PREFIX = "/api/v1"

# HTTP codes for domain errors, applied by the handler in server.py
ERROR_STATUS_MAP: dict[type[FalconError], HTTPStatus] = {
    ValidationError: HTTPStatus.UNPROCESSABLE_ENTITY,
    NotFoundError: HTTPStatus.NOT_FOUND,
    StorageError: HTTPStatus.SERVICE_UNAVAILABLE,
}

DEFAULT_ERROR_STATUS = HTTPStatus.INTERNAL_SERVER_ERROR

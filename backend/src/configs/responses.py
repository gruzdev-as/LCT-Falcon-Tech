from http import HTTPStatus
from typing import Any

from pydantic import BaseModel

from backend.src.configs.schemas import ErrorResponse, SearchPending


def describe(status: HTTPStatus, model: type[BaseModel], description: str) -> dict[int | str, dict[str, Any]]:
    """Declare one extra response for OpenAPI.

    The nested-dict shape is what ``responses=`` takes; it affects the generated
    schema only, never runtime behaviour. Wrapped here so routers pass a name.
    """
    return {int(status): {"model": model, "description": description}}


# Combined with | at the call site when an endpoint declares several
INVALID_INPUT = describe(HTTPStatus.UNPROCESSABLE_ENTITY, ErrorResponse, "Invalid image or bbox")
UNKNOWN_TASK = describe(HTTPStatus.NOT_FOUND, ErrorResponse, "Unknown or expired task")
STILL_PROCESSING = describe(HTTPStatus.ACCEPTED, SearchPending, "Task is still being processed")

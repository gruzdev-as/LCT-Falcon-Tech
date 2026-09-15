from fastapi import UploadFile
from pydantic import BaseModel, Field

from common.src.configs.constants import DEFAULT_TOP_K, MAX_TOP_K
from common.src.configs.schemas import TaskStatus


class SearchForm(BaseModel):
    """Multipart payload of a search submission."""

    file: UploadFile = Field(description="Image containing the vehicle")
    bbox: str = Field(description="Bounding box in pixels, as JSON: or [x,y,w,h]")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="How many to return")


class SearchAccepted(BaseModel):
    """Response to a submitted search: the task is queued, not finished."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING


class SearchPending(BaseModel):
    """Response while a task is still being processed."""

    task_id: str
    status: TaskStatus = TaskStatus.PROCESSING


class ErrorResponse(BaseModel):
    """Body returned for any domain error."""

    error: str = Field(description="Machine-readable error code.")
    message: str
    details: dict = Field(default_factory=dict)

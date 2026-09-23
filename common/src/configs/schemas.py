from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from common.src.configs.constants import DEFAULT_TOP_K, MAX_TOP_K, MIN_SIDE_PX


class TaskStatus(StrEnum):
    """Lifecycle of a search task."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class BBox(BaseModel):
    """Vehicle bounding box in absolute pixels of the source image."""

    x: float = Field(ge=0, description="Left edge, px")
    y: float = Field(ge=0, description="Top edge, px")
    width: float = Field(gt=0, description="Width, px")
    height: float = Field(gt=0, description="Height, px")

    @property
    def x2(self) -> float:
        """Right edge, px."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge, px."""
        return self.y + self.height

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Return integer ``(x1, y1, x2, y2)`` suitable for PIL/OpenCV cropping."""
        return int(self.x), int(self.y), int(round(self.x2)), int(round(self.y2))

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BBox":
        """Build a box from its corners."""
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def clamp(self, image_width: int, image_height: int) -> "BBox":
        """Trim the box to the image bounds.

        Args:
            image_width: real width of the decoded image, px.
            image_height: real height of the decoded image, px.

        Returns:
            A box guaranteed to lie inside the image.

        Raises:
            ValueError: the box degenerates once trimmed.
        """
        x1 = max(0.0, min(self.x, image_width - 1))
        y1 = max(0.0, min(self.y, image_height - 1))
        x2 = max(x1 + 1, min(self.x2, float(image_width)))
        y2 = max(y1 + 1, min(self.y2, float(image_height)))

        clamped = BBox.from_xyxy(x1, y1, x2, y2)
        if clamped.width < MIN_SIDE_PX or clamped.height < MIN_SIDE_PX:
            msg = f"bbox is smaller than {MIN_SIDE_PX}px per side after clamping to {image_width}x{image_height}"
            raise ValueError(msg)
        return clamped

    @model_validator(mode="after")
    def _check_side(self) -> "BBox":
        if self.width < MIN_SIDE_PX or self.height < MIN_SIDE_PX:
            msg = f"bbox side must be at least {MIN_SIDE_PX}px"
            raise ValueError(msg)
        return self


class EmbeddingTask(BaseModel):
    """Work item published by the backend onto the task stream."""

    task_id: str
    image_path: str
    """Path relative to the shared image directory. Inference reads the file itself."""

    bbox: BBox
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Candidate(BaseModel):
    """One gallery match returned by the vector search."""

    image_id: str
    score: float = Field(description="Similarity in [0, 1]; higher is closer.")
    rank: int = Field(ge=1)
    image_path: str | None = None
    image_url: str | None = Field(default=None, description="URL the frontend can load the original from.")
    vehicle_id: str | None = None
    camera_id: str | None = None


class SearchResult(BaseModel):
    """Finished search, published by inference onto the result stream."""

    task_id: str
    status: TaskStatus
    candidates: list[Candidate] = Field(default_factory=list)
    """Sorted by descending score. Empty when ``rejected`` is set."""

    top_score: float | None = None
    rejected: bool = Field(default=False, description="True when the best score fell below the threshold")
    model_name: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

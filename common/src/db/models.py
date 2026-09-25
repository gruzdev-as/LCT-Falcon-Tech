from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from common.src.configs.schemas import TaskStatus
from common.src.db.base import Base

_TASK_STATUS = Enum(
    TaskStatus,
    name="task_status",
    native_enum=False,
    length=16,
    # Without this SQLAlchemy stores the member name ("PENDING"), and the column stops
    # matching the value the Redis marker and the API carry.
    values_callable=lambda members: [member.value for member in members],
)


class GalleryImage(Base):
    """One indexed gallery image: the metadata half of a Qdrant point."""

    __tablename__ = "gallery_images"
    __table_args__ = (
        Index("ix_gallery_images_vehicle_id", "vehicle_id"),
        Index("ix_gallery_images_bundle_version", "bundle_version"),
    )

    image_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    image_path: Mapped[str] = mapped_column(String(1024))
    """Object key in S3, relative to the bucket."""

    vehicle_id: Mapped[str | None] = mapped_column(String(128), default=None)
    camera_id: Mapped[str | None] = mapped_column(String(128), default=None)
    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)

    bundle_version: Mapped[str] = mapped_column(String(128))
    """Which artifact bundle indexed this row; lets a stale gallery be spotted."""

    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )


class SearchQuery(Base):
    """One accepted search request and, once it finishes, its outcome."""

    __tablename__ = "search_queries"
    __table_args__ = (Index("ix_search_queries_created_at", "created_at"),)

    task_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    """``uuid4().hex`` minted by the backend; the id the client polls."""

    image_path: Mapped[str] = mapped_column(String(1024))
    """Object key of the stored original, under the ``queries/`` prefix."""

    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_width: Mapped[float] = mapped_column(Float)
    bbox_height: Mapped[float] = mapped_column(Float)
    """The box after clamping to the real frame: absolute px, xywh from the top-left."""

    top_k: Mapped[int] = mapped_column(Integer)

    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    image_format: Mapped[str] = mapped_column(String(16))
    content_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)

    status: Mapped[TaskStatus] = mapped_column(_TASK_STATUS, default=TaskStatus.PENDING)
    """Goes ``pending`` -> ``done``/``failed``: the backend never observes ``processing``."""

    top_score: Mapped[float | None] = mapped_column(Float, default=None)
    rejected: Mapped[bool | None] = mapped_column(Boolean, default=None)
    candidate_count: Mapped[int | None] = mapped_column(Integer, default=None)
    model_name: Mapped[str | None] = mapped_column(String(128), default=None)
    latency_ms: Mapped[float | None] = mapped_column(Float, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    """Inference's clock, copied off the ``SearchResult``."""

    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    """When this backend recorded the outcome. NULL is the idempotency guard."""


class SearchCandidate(Base):
    """One ranked match a finished search returned.

    A historical fact, not a live pointer: the row records what the search answered at
    the time, even after the gallery is rebuilt from a different bundle. That is why
    ``image_id`` is deliberately not a foreign key into ``gallery_images``.
    """

    __tablename__ = "search_candidates"
    __table_args__ = (Index("ix_search_candidates_image_id", "image_id"),)

    task_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("search_queries.task_id", ondelete="CASCADE"),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    """1-based position. Together with ``task_id`` it is the primary key, which is what
    makes re-running the finalize step a no-op instead of a duplicate."""

    image_id: Mapped[str] = mapped_column(String(128))
    """Gallery id, read off the Qdrant payload."""

    score: Mapped[float] = mapped_column(Float)
    image_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    vehicle_id: Mapped[str | None] = mapped_column(String(128), default=None)
    camera_id: Mapped[str | None] = mapped_column(String(128), default=None)

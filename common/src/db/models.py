from datetime import UTC, datetime

from sqlalchemy import DateTime, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from common.src.db.base import Base


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

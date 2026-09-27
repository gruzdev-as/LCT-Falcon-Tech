import logging
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from common.src.db.models import GalleryImage
from init.src.gallery import Gallery
from init.src.steps.images import object_key

logger = logging.getLogger(__name__)

_CHUNK = 1000


async def count_rows(session: AsyncSession, gallery_version: str) -> int:
    """How many gallery rows this version of the manifest already wrote."""
    statement = select(func.count()).select_from(GalleryImage).where(GalleryImage.gallery_version == gallery_version)
    return int(await session.scalar(statement) or 0)


async def sync_gallery(session: AsyncSession, gallery: Gallery, *, force: bool = False) -> int:
    """Upsert the manifest into ``gallery_images``.

    Args:
        session: open session; not committed here.
        gallery: validated gallery.
        force: write every row even when the count already matches.

    Returns:
        How many rows were written by this call.
    """
    existing = await count_rows(session, gallery.version)
    if not force and existing == gallery.count:
        logger.info("Gallery metadata already holds %d rows for gallery %s, skipping", existing, gallery.version)
        return 0

    indexed_at = datetime.now(UTC)
    rows = [
        {
            "image_id": entry.image_id,
            "image_path": object_key(entry),
            "vehicle_id": entry.vehicle_id,
            "camera_id": entry.camera_id,
            "bbox_x": entry.bbox.x,
            "bbox_y": entry.bbox.y,
            "bbox_width": entry.bbox.width,
            "bbox_height": entry.bbox.height,
            "gallery_version": gallery.version,
            "indexed_at": indexed_at,
        }
        for entry in gallery.entries
    ]

    for start in range(0, len(rows), _CHUNK):
        statement = insert(GalleryImage).values(rows[start : start + _CHUNK])
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[GalleryImage.image_id],
                set_={
                    "image_path": statement.excluded.image_path,
                    "vehicle_id": statement.excluded.vehicle_id,
                    "camera_id": statement.excluded.camera_id,
                    "bbox_x": statement.excluded.bbox_x,
                    "bbox_y": statement.excluded.bbox_y,
                    "bbox_width": statement.excluded.bbox_width,
                    "bbox_height": statement.excluded.bbox_height,
                    "gallery_version": statement.excluded.gallery_version,
                    "indexed_at": statement.excluded.indexed_at,
                },
            )
        )

    logger.info("Wrote %d gallery metadata rows for gallery %s", len(rows), gallery.version)
    return len(rows)

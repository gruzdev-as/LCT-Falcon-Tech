import logging
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from common.src.db.models import GalleryImage
from init.src.bundle import Bundle
from init.src.steps.images import object_key

logger = logging.getLogger(__name__)

_CHUNK = 1000


async def count_rows(session: AsyncSession, bundle_version: str) -> int:
    """How many gallery rows this bundle version already wrote."""
    statement = select(func.count()).select_from(GalleryImage).where(GalleryImage.bundle_version == bundle_version)
    return int(await session.scalar(statement) or 0)


async def sync_gallery(session: AsyncSession, bundle: Bundle, *, force: bool = False) -> int:
    """Upsert the manifest into ``gallery_images``.

    Args:
        session: open session; not committed here.
        bundle: validated bundle.
        force: write every row even when the count already matches.

    Returns:
        How many rows were written by this call.
    """
    existing = await count_rows(session, bundle.version)
    if not force and existing == bundle.count:
        logger.info("Gallery metadata already holds %d rows for bundle %s, skipping", existing, bundle.version)
        return 0

    indexed_at = datetime.now(UTC)
    rows = [
        {
            "image_id": entry.image_id,
            "image_path": object_key(entry),
            "vehicle_id": entry.vehicle_id,
            "camera_id": entry.camera_id,
            "bundle_version": bundle.version,
            "indexed_at": indexed_at,
        }
        for entry in bundle.entries
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
                    "bundle_version": statement.excluded.bundle_version,
                    "indexed_at": statement.excluded.indexed_at,
                },
            )
        )

    logger.info("Wrote %d gallery metadata rows for bundle %s", len(rows), bundle.version)
    return len(rows)

import logging
import uuid

from qdrant_client.models import PointStruct

from common.src.configs.constants import GALLERY_COLLECTION
from common.src.qdrant import client as qdrant
from init.src.bundle import Bundle, GalleryEntry
from init.src.configs.constants import POINT_NAMESPACE
from init.src.steps.images import object_key

logger = logging.getLogger(__name__)


def point_id(image_id: str) -> str:
    """Map a dataset image id onto a Qdrant point id."""
    return str(uuid.uuid5(POINT_NAMESPACE, image_id))


def build_point(entry: GalleryEntry, vector: list[float]) -> PointStruct:
    """Assemble one gallery point with the payload the search result reads."""
    return PointStruct(
        id=point_id(entry.image_id),
        vector=vector,
        payload={
            "image_id": entry.image_id,
            "image_path": object_key(entry),
            # The gallery stores whole frames, so a client cannot show a candidate without
            # this. It rides in the payload and not in Postgres because the request path
            # reads the payload; hitting the database there is against the backend's rules.
            "bbox": entry.bbox.model_dump() if entry.bbox else None,
            "vehicle_id": entry.vehicle_id,
            "camera_id": entry.camera_id,
        },
    )


async def index_gallery(bundle: Bundle, *, batch_size: int, force: bool = False) -> int:
    """Load the bundle's vectors into the gallery collection.

    Args:
        bundle: validated bundle.
        batch_size: points per upsert call.
        force: rebuild from scratch regardless of what the collection holds.

    Returns:
        How many points were written by this call.
    """
    indexed = await qdrant.count_points(GALLERY_COLLECTION)

    if not force and indexed == bundle.count:
        logger.info("Gallery collection already holds %d points, skipping", indexed)
        return 0

    if indexed:
        logger.info("Rebuilding the gallery collection (%d existing points)", indexed)
        await qdrant.recreate_collection(GALLERY_COLLECTION, bundle.embedding_dim)
    else:
        await qdrant.ensure_collection(GALLERY_COLLECTION, bundle.embedding_dim)

    logger.info("Indexing %d vectors (dim=%d, model=%s)", bundle.count, bundle.embedding_dim, bundle.model_name)
    vectors = bundle.embeddings.tolist()
    for start in range(0, bundle.count, batch_size):
        stop = start + batch_size
        batch = zip(bundle.entries[start:stop], vectors[start:stop], strict=True)
        await qdrant.upsert_points(GALLERY_COLLECTION, [build_point(entry, vector) for entry, vector in batch])
        logger.debug("Indexed %d/%d", min(stop, bundle.count), bundle.count)

    logger.info("Gallery collection now holds %d points", await qdrant.count_points(GALLERY_COLLECTION))
    return bundle.count

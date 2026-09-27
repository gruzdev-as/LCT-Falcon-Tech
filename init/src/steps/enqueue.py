import logging
import uuid
from dataclasses import dataclass

from common.src.configs.constants import (
    GALLERY_COLLECTION,
    GALLERY_FAILED_KEY,
    GALLERY_STATE_KEY,
    GALLERY_STATE_MODEL,
    GALLERY_STATE_TOTAL,
    GALLERY_STREAM,
    INFERENCE_GROUP,
)
from common.src.configs.schemas import GalleryTask
from common.src.qdrant import client as qdrant
from common.src.qdrant.gallery import MODEL_VERSION_FIELD, point_id
from common.src.redis.client import delete_stream, get_redis, publish, stream_backlog
from init.src.gallery import Gallery
from init.src.steps.images import object_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Schedule:
    """What one run asked the workers to do."""

    queued: int
    """Gallery images published for embedding by this run."""

    removed: int
    """Points dropped because they came from other weights or left the manifest."""

    already_indexed: int


async def schedule_gallery(gallery: Gallery, model_version: str, *, force: bool = False) -> Schedule:
    """Make the collection match the gallery and queue whatever it lacks for the workers.

    Points from other weights, or of images the manifest no longer lists, are deleted first.

    Args:
        gallery: validated gallery; its images are already in object storage.
        model_version: sha256 of the weights the workers serve.
        force: drop the collection and the queued tasks, and embed everything again.

    Returns:
        What was queued and removed.
    """
    if force:
        await qdrant.delete_collection(GALLERY_COLLECTION)
        await delete_stream(GALLERY_STREAM)

    indexed = await qdrant.payload_values(GALLERY_COLLECTION, MODEL_VERSION_FIELD)
    wanted = {point_id(entry.image_id): entry for entry in gallery.entries}

    stale = [pid for pid, version in indexed.items() if pid not in wanted or version != model_version]
    await qdrant.delete_points(GALLERY_COLLECTION, stale)
    if stale:
        logger.info("Removed %d gallery points from other weights or no longer in the manifest", len(stale))

    missing = [entry for pid, entry in wanted.items() if indexed.get(pid) != model_version]
    done = gallery.count - len(missing)
    await _write_state(model_version, gallery.count)

    if not missing:
        logger.info("Gallery already indexed: %d images with weights %s", done, model_version[:12])
        return Schedule(queued=0, removed=len(stale), already_indexed=done)

    backlog = await stream_backlog(GALLERY_STREAM, INFERENCE_GROUP)
    if backlog:
        # A previous run queued them and the workers have not finished; queueing again
        # would only embed the same images twice.
        logger.info("%d gallery tasks are still queued from a previous run, not queueing again", backlog)
        return Schedule(queued=0, removed=len(stale), already_indexed=done)

    for entry in missing:
        await publish(
            GALLERY_STREAM,
            GalleryTask(
                task_id=uuid.uuid4().hex,
                image_id=entry.image_id,
                image_path=object_key(entry),
                bbox=entry.bbox,
                vehicle_id=entry.vehicle_id,
                camera_id=entry.camera_id,
                model_version=model_version,
            ),
        )
    logger.info(
        "Queued %d of %d gallery images for embedding on the inference workers (weights %s)",
        len(missing),
        gallery.count,
        model_version[:12],
    )
    return Schedule(queued=len(missing), removed=len(stale), already_indexed=done)


async def _write_state(model_version: str, total: int) -> None:
    """Tell gallery-ready what a finished gallery looks like, and forget the last run's failures."""
    async with get_redis().pipeline(transaction=True) as pipe:
        pipe.hset(GALLERY_STATE_KEY, mapping={GALLERY_STATE_MODEL: model_version, GALLERY_STATE_TOTAL: total})
        pipe.delete(GALLERY_FAILED_KEY)
        await pipe.execute()

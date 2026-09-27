import logging

from common.src.db.session import close_engine, create_schema, session_scope
from common.src.qdrant.client import close_qdrant
from common.src.redis.client import close_redis
from init.src.artifacts import fetch_all, model_version
from init.src.configs.settings import InitSettings
from init.src.gallery import load_gallery
from init.src.steps import enqueue, images, metadata

logger = logging.getLogger(__name__)


async def bootstrap(settings: InitSettings) -> None:
    """Bring an empty stack to the point where the workers can build the gallery.

    Init embeds nothing: it stores the mounted gallery and queues it for the inference
    workers, and gallery-ready waits for them to finish.

    Raises:
        StorageError: a download failed or a store is unreachable.
        ValidationError: the mounted gallery is missing or inconsistent, or there is no model.
    """
    settings.weights_dir.mkdir(parents=True, exist_ok=True)

    await fetch_all(settings)
    version = await model_version(settings)
    await create_schema()

    gallery = load_gallery(settings.gallery_dir)
    logger.info("Gallery %s: %d images from %s", gallery.version, gallery.count, settings.gallery_dir)

    uploaded = await images.upload_gallery(gallery, concurrency=settings.upload_concurrency, force=settings.force)
    async with session_scope() as session:
        rows = await metadata.sync_gallery(session, gallery, force=settings.force)
    schedule = await enqueue.schedule_gallery(gallery, version, force=settings.force)

    logger.info(
        "Bootstrap finished: uploaded=%d rows=%d queued=%d removed=%d already_indexed=%d",
        uploaded,
        rows,
        schedule.queued,
        schedule.removed,
        schedule.already_indexed,
    )


async def shutdown() -> None:
    """Release the clients the steps opened."""
    await close_qdrant()
    await close_redis()
    await close_engine()

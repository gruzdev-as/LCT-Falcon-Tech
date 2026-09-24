import logging

from common.src.db.session import close_engine, create_schema, session_scope
from common.src.qdrant.client import close_qdrant
from init.src.artifacts import fetch_all
from init.src.bundle import Bundle, bundle_present, load_bundle
from init.src.configs.settings import InitSettings
from init.src.steps import images, metadata, vectors

logger = logging.getLogger(__name__)


async def bootstrap(settings: InitSettings) -> None:
    """Bring an empty stack to a state where a search can return something.

    Raises:
        StorageError: a download failed or a store is unreachable.
        ValidationError: the artifact bundle is inconsistent.
    """
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.weights_dir.mkdir(parents=True, exist_ok=True)

    await fetch_all(settings)
    await create_schema()

    bundle = _load_or_skip(settings)
    if bundle is None:
        return

    logger.info(
        "Bundle %s: %d images, dim=%d, model=%s",
        bundle.version,
        bundle.count,
        bundle.embedding_dim,
        bundle.model_name,
    )
    logger.info("Set INFERENCE_EMBEDDING_DIM=%d so the workers match this gallery", bundle.embedding_dim)

    uploaded = await images.upload_gallery(bundle, concurrency=settings.upload_concurrency, force=settings.force)
    indexed = await vectors.index_gallery(bundle, batch_size=settings.upsert_batch, force=settings.force)
    async with session_scope() as session:
        rows = await metadata.sync_gallery(session, bundle, force=settings.force)

    logger.info("Bootstrap finished: uploaded=%d indexed=%d rows=%d", uploaded, indexed, rows)
    if not settings.force:
        logger.info("Replaced an artifact? Rerun with --force, the gates only check what is already there")


def _load_or_skip(settings: InitSettings) -> Bundle | None:
    """Load the bundle, or report that there is nothing to index yet."""
    if bundle_present(settings.artifacts_dir):
        return load_bundle(settings.artifacts_dir)

    logger.warning(
        "No bundle in %s: the gallery stays empty and every search will answer rejected. "
        "Set INIT_VECTORS_URL or unpack a bundle there.",
        settings.artifacts_dir,
    )
    return None


async def shutdown() -> None:
    """Release the clients the steps opened."""
    await close_qdrant()
    await close_engine()

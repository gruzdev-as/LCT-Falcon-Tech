import logging
import time
from dataclasses import dataclass

import anyio.to_thread

from common.src.configs.constants import GALLERY_COLLECTION, GALLERY_FAILED_KEY
from common.src.configs.schemas import GalleryTask
from common.src.exceptions import NotFoundError, ValidationError
from common.src.qdrant import client as qdrant
from common.src.qdrant.gallery import gallery_point
from common.src.redis.client import get_redis
from common.src.storage import client as storage
from inference.src.models.base import Embedder
from inference.src.pipeline.embed import embed_vector

logger = logging.getLogger(__name__)


class ForeignModelError(RuntimeError):
    """The task was queued for other weights than this worker serves."""


@dataclass(frozen=True)
class GalleryIndexer:
    """Embed one gallery image and index it, the same way a search embeds its query."""

    embedder: Embedder

    async def index(self, task: GalleryTask) -> bool:
        """Embed the image and upsert its point.

        Returns:
            True when the point was written, False when the image is broken and skipped.

        Raises:
            ForeignModelError: the task is for other weights; it stays unacked.
            StorageError: object storage is unavailable.
            Exception: Qdrant or the model failed in a way unrelated to this image.
        """
        if task.model_version != self.embedder.version:
            msg = f"task wants weights {task.model_version[:12]}, this worker serves {self.embedder.version[:12]}"
            raise ForeignModelError(msg)

        started = time.monotonic()
        try:
            data = await storage.load(task.image_path)
            vector = await anyio.to_thread.run_sync(embed_vector, self.embedder, data, task.bbox)
        except (ValidationError, NotFoundError) as exc:
            await self.skip(task, exc.message)
            return False

        point = gallery_point(
            image_id=task.image_id,
            image_path=task.image_path,
            bbox=task.bbox,
            vehicle_id=task.vehicle_id,
            camera_id=task.camera_id,
            model_version=self.embedder.version,
            vector=vector.tolist(),
        )
        # init may have dropped the collection for a rebuild while this worker was running.
        await qdrant.ensure_collection(GALLERY_COLLECTION, self.embedder.dim)
        await qdrant.upsert_points(GALLERY_COLLECTION, [point])
        logger.info(
            "task=%s stage=gallery image_id=%s indexed latency_ms=%.0f",
            task.task_id,
            task.image_id,
            (time.monotonic() - started) * 1000,
        )
        return True

    async def skip(self, task: GalleryTask, reason: str) -> None:
        """Record an image the gallery will go without, so the wait for it can end."""
        await get_redis().sadd(GALLERY_FAILED_KEY, task.image_id)
        logger.warning("task=%s stage=gallery image_id=%s skipped: %s", task.task_id, task.image_id, reason)

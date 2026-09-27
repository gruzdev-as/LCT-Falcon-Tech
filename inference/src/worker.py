import logging
import os
import socket
import time

import anyio

from common.src.configs.constants import GALLERY_COLLECTION, GALLERY_STREAM, INFERENCE_GROUP, TASK_STREAM
from common.src.configs.schemas import EmbeddingTask, GalleryTask
from common.src.qdrant.client import ensure_collection
from common.src.redis.client import ack, claim_one, init_redis_streams, read_one
from inference.src.configs.constants import BACKOFF_SECONDS, POISON_ERROR
from inference.src.configs.settings import InferenceSettings
from inference.src.gallery import ForeignModelError, GalleryIndexer
from inference.src.processor import Processor
from inference.src.results import mark_processing, write_result

logger = logging.getLogger(__name__)

type Entry = tuple[str, EmbeddingTask]
type GalleryEntry = tuple[str, GalleryTask]


class InferenceWorker:
    """One member of the ``inference-workers`` consumer group."""

    def __init__(self, processor: Processor, settings: InferenceSettings, consumer: str | None = None) -> None:
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self._processor = processor
        self._gallery = GalleryIndexer(processor.embedder)
        self._settings = settings
        self._stopping = False
        self._next_claim_at = 0.0
        self._next_gallery_claim_at = 0.0

    def stop(self) -> None:
        """Ask the loop to exit after the task in flight is written and acked."""
        self._stopping = True

    async def start(self) -> None:
        """Prepare the stream group and the gallery collection, idempotently."""
        await init_redis_streams()
        await ensure_collection(GALLERY_COLLECTION, self._processor.embedder.dim)
        logger.info("Inference worker %s ready (model=%s)", self.consumer, self._processor.embedder.name)

    async def run(self) -> None:
        """Consume until ``stop`` is called."""
        await self.start()
        while not self._stopping:
            try:
                await self.run_once()
            except Exception:
                logger.exception("Iteration failed, the task stays pending for a retry")
                await anyio.sleep(BACKOFF_SECONDS)
                continue
            self._settings.heartbeat_path.touch()
        logger.info("Inference worker %s stopped", self.consumer)

    async def run_once(self) -> bool:
        """Do one unit of work: a search if there is one, else one gallery image, else wait for a search.

        A search that arrives mid-gallery waits for at most one embedding.

        Returns:
            Whether a search result was written or a gallery point indexed.
        """
        entry = await self._reclaim() or await self._read(block_ms=None)
        if entry is not None:
            return await self._answer(entry)

        gallery = await self._reclaim_gallery() or await self._read_gallery()
        if gallery is not None:
            return await self._index(gallery)

        entry = await self._read(block_ms=self._settings.block_ms)
        return await self._answer(entry) if entry is not None else False

    async def _answer(self, entry: Entry) -> bool:
        """Answer one search and ack it once the result is written."""
        message_id, task = entry
        if not await mark_processing(task.task_id):
            logger.info("task=%s stage=processing skipped, expired or already answered", task.task_id)
            await ack(TASK_STREAM, INFERENCE_GROUP, message_id)
            return False

        result = await self._processor.process(task)
        await write_result(result)
        await ack(TASK_STREAM, INFERENCE_GROUP, message_id)
        logger.info(
            "task=%s stage=result status=%s rejected=%s top_score=%s candidates=%d latency_ms=%.0f",
            task.task_id,
            result.status,
            result.rejected,
            f"{result.top_score:.3f}" if result.top_score is not None else "-",
            len(result.candidates),
            result.latency_ms or 0.0,
        )
        return True

    async def _index(self, entry: GalleryEntry) -> bool:
        """Index one gallery image and ack it; a task for other weights stays pending."""
        message_id, task = entry
        try:
            indexed = await self._gallery.index(task)
        except ForeignModelError as exc:
            # Unacked, it is reclaimed later and ends as poison unless a worker with
            # the right weights takes it first.
            logger.error("task=%s stage=gallery image_id=%s refused: %s", task.task_id, task.image_id, exc)  # noqa: TRY400
            await anyio.sleep(BACKOFF_SECONDS)
            return False
        await ack(GALLERY_STREAM, INFERENCE_GROUP, message_id)
        return indexed

    async def _read(self, *, block_ms: int | None) -> Entry | None:
        return await read_one(TASK_STREAM, INFERENCE_GROUP, self.consumer, EmbeddingTask, block_ms=block_ms)

    async def _read_gallery(self) -> GalleryEntry | None:
        return await read_one(GALLERY_STREAM, INFERENCE_GROUP, self.consumer, GalleryTask, block_ms=None)

    async def _reclaim(self) -> Entry | None:
        """Claim a task abandoned by a dead replica; fail it if it keeps crashing workers."""
        if time.monotonic() < self._next_claim_at:
            return None

        claimed = await claim_one(
            TASK_STREAM,
            INFERENCE_GROUP,
            self.consumer,
            EmbeddingTask,
            min_idle_ms=self._settings.claim_idle_ms,
        )
        if claimed is None:
            # Nothing abandoned: check again after the interval.
            self._next_claim_at = time.monotonic() + self._settings.claim_interval_s
            return None

        message_id, task, deliveries = claimed
        if deliveries <= self._settings.max_deliveries:
            logger.warning("task=%s stage=processing reclaimed, delivery %d", task.task_id, deliveries)
            return message_id, task

        # This claim is itself a delivery, so the task was attempted one time fewer
        logger.error("task=%s stage=processing poison after %d attempts", task.task_id, deliveries - 1)
        if await mark_processing(task.task_id):
            await write_result(self._processor.failed(task, POISON_ERROR))
        await ack(TASK_STREAM, INFERENCE_GROUP, message_id)
        return None

    async def _reclaim_gallery(self) -> GalleryEntry | None:
        """Claim a gallery image a dead replica held; skip it if it keeps crashing workers."""
        if time.monotonic() < self._next_gallery_claim_at:
            return None

        claimed = await claim_one(
            GALLERY_STREAM,
            INFERENCE_GROUP,
            self.consumer,
            GalleryTask,
            min_idle_ms=self._settings.claim_idle_ms,
        )
        if claimed is None:
            self._next_gallery_claim_at = time.monotonic() + self._settings.claim_interval_s
            return None

        message_id, task, deliveries = claimed
        if deliveries <= self._settings.max_deliveries:
            logger.warning("task=%s stage=gallery reclaimed, delivery %d", task.task_id, deliveries)
            return message_id, task

        await self._gallery.skip(task, f"poison after {deliveries - 1} attempts")
        await ack(GALLERY_STREAM, INFERENCE_GROUP, message_id)
        return None

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import anyio.to_thread
import numpy as np
from qdrant_client.models import ScoredPoint

from common.src.configs.constants import GALLERY_COLLECTION
from common.src.configs.schemas import BBox, EmbeddingTask, SearchResult, TaskStatus
from common.src.exceptions import NotFoundError, ValidationError
from common.src.qdrant import client as qdrant
from common.src.storage import client as storage
from inference.src.models.base import Embedder
from inference.src.pipeline.crop import crop_vehicle
from inference.src.pipeline.postprocess import l2_normalize, rank_candidates
from inference.src.pipeline.preprocess import preprocess

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Processor:
    """Processing, Analysis and Result stages for one task."""

    embedder: Embedder
    reject_threshold: float

    async def process(self, task: EmbeddingTask) -> SearchResult:
        """Crop, embed, search and rank one task.

        Raises:
            StorageError: object storage is unavailable.
            Exception: Qdrant or the model failed in a way unrelated to this task.
        """
        try:
            data = await storage.load(task.image_path)
            vector = await anyio.to_thread.run_sync(self._embed, data, task.bbox)
        except (ValidationError, NotFoundError) as exc:
            logger.warning("task=%s stage=processing failed: %s", task.task_id, exc.message)
            return self.failed(task, exc.message)

        hits = await qdrant.search(GALLERY_COLLECTION, vector.tolist(), task.top_k)
        return self._rank(task, hits)

    def failed(self, task: EmbeddingTask, error: str) -> SearchResult:
        """Build the result for a task that cannot be completed."""
        return SearchResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=error,
            model_name=self.embedder.name,
            latency_ms=_latency_ms(task),
        )

    def _embed(self, data: bytes, bbox: BBox) -> np.ndarray:
        """Crop, preprocess and embed in one blocking call, run in a worker thread."""
        crop = preprocess(crop_vehicle(data, bbox), self.embedder.input_size)
        vector = l2_normalize(self.embedder.embed(crop))
        if vector.shape != (self.embedder.dim,):
            msg = f"embedder returned shape {vector.shape}, expected ({self.embedder.dim},)"
            raise RuntimeError(msg)
        return vector

    def _rank(self, task: EmbeddingTask, hits: Sequence[ScoredPoint]) -> SearchResult:
        candidates, top_score, rejected = rank_candidates(
            hits, threshold=self.reject_threshold, sign_url=storage.url_for
        )
        return SearchResult(
            task_id=task.task_id,
            status=TaskStatus.DONE,
            candidates=candidates,
            top_score=top_score,
            rejected=rejected,
            model_name=self.embedder.name,
            latency_ms=_latency_ms(task),
        )


def _latency_ms(task: EmbeddingTask) -> float:
    """Time since the backend queued the task: queueing plus processing."""
    return (datetime.now(UTC) - task.created_at).total_seconds() * 1000

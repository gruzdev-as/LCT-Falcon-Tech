import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import anyio.to_thread
from qdrant_client.models import ScoredPoint

from common.src.configs.constants import GALLERY_COLLECTION
from common.src.configs.schemas import EmbeddingTask, SearchResult, TaskStatus
from common.src.exceptions import NotFoundError, ValidationError
from common.src.qdrant import client as qdrant
from common.src.storage import client as storage
from inference.src.models.base import Embedder
from inference.src.models.refusal import Refusal
from inference.src.pipeline.embed import embed_vector
from inference.src.pipeline.postprocess import rank_candidates

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Processor:
    """Processing, Analysis and Result stages for one task."""

    embedder: Embedder
    refusal: Refusal

    async def process(self, task: EmbeddingTask) -> SearchResult:
        """Crop, embed, search and rank one task.

        Raises:
            StorageError: object storage is unavailable.
            Exception: Qdrant or the model failed in a way unrelated to this task.
        """
        try:
            data = await storage.load(task.image_path)
            vector = await anyio.to_thread.run_sync(embed_vector, self.embedder, data, task.bbox)
        except (ValidationError, NotFoundError) as exc:
            logger.warning("task=%s stage=processing failed: %s", task.task_id, exc.message)
            return self.failed(task, exc.message)

        # The refusal may need more neighbours than the caller asked to see.
        limit = max(task.top_k, self.refusal.neighbours)
        hits = await qdrant.search(GALLERY_COLLECTION, vector.tolist(), limit, with_vectors=self.refusal.needs_vectors)
        accepted = bool(hits) and await anyio.to_thread.run_sync(
            self.refusal.accept, vector, hits[: self.refusal.neighbours]
        )
        logger.info(
            "task=%s stage=analysis refusal=%s accepted=%s hits=%d",
            task.task_id,
            self.refusal.name,
            accepted,
            len(hits),
        )
        return self._rank(task, hits[: task.top_k], accepted=accepted)

    def failed(self, task: EmbeddingTask, error: str) -> SearchResult:
        """Build the result for a task that cannot be completed."""
        return SearchResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=error,
            model_name=self.embedder.name,
            latency_ms=_latency_ms(task),
        )

    def _rank(self, task: EmbeddingTask, hits: Sequence[ScoredPoint], *, accepted: bool) -> SearchResult:
        candidates, top_score, rejected = rank_candidates(
            hits, accepted=accepted, min_score=self.refusal.min_score, sign_url=storage.url_for
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

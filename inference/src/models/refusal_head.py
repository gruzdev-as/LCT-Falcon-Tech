import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from qdrant_client.models import ScoredPoint
from refusal import load_boosting, refusal_accept

from inference.src.configs.refusal import RefusalPreset

logger = logging.getLogger(__name__)


class HeadRefusal:
    """The refusal training serves: the CatBoost head, alone or together with the cosine threshold."""

    needs_vectors = True

    def __init__(self, preset: RefusalPreset, model_path: Path) -> None:
        """Load the CatBoost head once.

        Raises:
            ValueError: no model file at ``model_path``.
        """
        self.preset = preset
        self.name = preset.kind
        self.neighbours = preset.k
        self.min_score = preset.cosine_threshold if preset.cosine_threshold is not None else 0.0
        if preset.cosine_threshold is None:
            logger.warning("Refusal preset %s has no cosine_threshold: candidates are not filtered", preset.kind)
        # load_boosting raises ValueError itself when the file is missing.
        self._model = load_boosting(model_path)
        logger.info(
            "Loaded refusal head %s from %s: kind=%s k=%d cosine>=%s P>=%s",
            type(self._model).__name__,
            model_path,
            preset.kind,
            preset.k,
            preset.cosine_threshold,
            preset.model_threshold,
        )

    def accept(self, query: np.ndarray, hits: Sequence[ScoredPoint]) -> bool:
        """Build training's features from the query and its neighbours, and ask the head."""
        gallery = np.asarray([_vector(hit) for hit in hits[: self.neighbours]], dtype=np.float32)
        decision = refusal_accept(
            self.preset.kind,
            query,
            gallery,
            cosine_threshold=self.preset.cosine_threshold,
            model=self._model,
            model_threshold=self.preset.model_threshold,
            k=self.preset.k,
            with_embeddings=self.preset.with_embeddings,
        )
        return bool(decision[0])


def _vector(hit: ScoredPoint) -> list[float]:
    if not isinstance(hit.vector, list):
        msg = f"point {hit.id} came back without its vector; search with with_vectors=True"
        raise TypeError(msg)
    return hit.vector

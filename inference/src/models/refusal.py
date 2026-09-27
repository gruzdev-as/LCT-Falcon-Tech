from collections.abc import Sequence
from typing import Protocol

import numpy as np
from qdrant_client.models import ScoredPoint


class Refusal(Protocol):
    """Decides whether the gallery holds the query's vehicle at all.

    It sees the query vector and its nearest gallery points, never the whole gallery:
    the Analysis stage stays Qdrant's job.
    """

    name: str
    """Logged with every decision, e.g. ``ensemble`` or ``cosine``."""

    neighbours: int
    """How many nearest gallery points ``accept`` needs; the search asks for at least this many."""

    needs_vectors: bool
    """Whether those points must come with their stored vectors, not only scores."""

    min_score: float
    """Floor every returned candidate must clear, so an accepted search is not padded with weak matches."""

    def accept(self, query: np.ndarray, hits: Sequence[ScoredPoint]) -> bool:
        """Return True when the best hits are a real match, False to answer rejected.

        Blocking; the processor calls it off the event loop.

        Args:
            query: the L2-normalized query vector.
            hits: nonempty, sorted by descending score, at most ``neighbours`` long.
        """
        ...


class CosineRefusal:
    """Reject when the best cosine score is below a threshold."""

    name = "cosine"
    neighbours = 1
    needs_vectors = False

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.min_score = threshold

    def accept(self, query: np.ndarray, hits: Sequence[ScoredPoint]) -> bool:  # noqa: ARG002  # protocol
        """Compare the top score, which Qdrant already computed as a cosine."""
        return hits[0].score >= self.threshold

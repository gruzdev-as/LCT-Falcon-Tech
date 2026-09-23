from collections.abc import Callable, Sequence

import numpy as np
from qdrant_client.models import ScoredPoint

from common.src.configs.schemas import Candidate

# Guards the division for an all-zero vector, which then stays zero and matches nothing
_EPS = 1e-12


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale a vector (or every row of a matrix) to unit length.

    After this, cosine and dot product rank identically.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, _EPS)


def to_similarity(cosine: float) -> float:
    """Map a cosine score onto the [0, 1] range the API promises."""
    return min(1.0, max(0.0, cosine))


def rank_candidates(
    points: Sequence[ScoredPoint],
    *,
    threshold: float,
    sign_url: Callable[[str], str],
) -> tuple[list[Candidate], float | None, bool]:
    """Turn raw search hits into the Result stage output.

    Args:
        points: gallery hits for one query.
        threshold: minimum score of the best candidate.
        sign_url: builds a browser-loadable link for a stored image path.

    Returns:
        ``(candidates, top_score, rejected)``. Candidates are sorted by descending
        score; when the best one falls below the threshold — or there are no hits —
        the list is empty and ``rejected`` is set. Weak matches are never returned.
    """
    ordered = sorted(points, key=lambda point: point.score, reverse=True)
    top_score = to_similarity(ordered[0].score) if ordered else None
    if top_score is None or top_score < threshold:
        return [], top_score, True

    candidates = []
    for rank, point in enumerate(ordered, start=1):
        payload = point.payload or {}
        image_path = payload.get("image_path")
        candidates.append(
            Candidate(
                image_id=str(point.id),
                score=to_similarity(point.score),
                rank=rank,
                image_path=image_path,
                image_url=sign_url(image_path) if image_path else None,
                vehicle_id=payload.get("vehicle_id"),
                camera_id=payload.get("camera_id"),
            )
        )
    return candidates, top_score, False

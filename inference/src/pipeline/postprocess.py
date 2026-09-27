from collections.abc import Callable, Sequence

import numpy as np
from qdrant_client.models import ScoredPoint

from common.src.configs.schemas import BBox, Candidate

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
    accepted: bool,
    min_score: float,
    sign_url: Callable[[str], str],
) -> tuple[list[Candidate], float | None, bool]:
    """Turn raw search hits into the Result stage output.

    Args:
        points: gallery hits for one query.
        accepted: the refusal decision; False answers rejected however good the hits look.
        min_score: every returned candidate must score at least this.
        sign_url: builds a browser-loadable link for a stored image path.

    Returns:
        (candidates, top_score, rejected)
    """
    ordered = sorted(points, key=lambda point: point.score, reverse=True)
    top_score = to_similarity(ordered[0].score) if ordered else None

    kept = [point for point in ordered if point.score >= min_score]
    if not accepted or not kept:
        return [], top_score, True
    ordered = kept

    candidates = []
    for rank, point in enumerate(ordered, start=1):
        payload = point.payload or {}
        image_path = payload.get("image_path")
        bbox = payload.get("bbox")
        candidates.append(
            Candidate(
                image_id=str(payload.get("image_id") or point.id),
                score=to_similarity(point.score),
                rank=rank,
                image_path=image_path,
                image_url=sign_url(image_path) if image_path else None,
                # A gallery indexed before the box existed simply has no key here.
                bbox=BBox(**bbox) if bbox else None,
                vehicle_id=payload.get("vehicle_id"),
                camera_id=payload.get("camera_id"),
            )
        )
    return candidates, top_score, False

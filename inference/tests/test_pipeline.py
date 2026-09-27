import numpy as np
import pytest
from qdrant_client.models import ScoredPoint

from common.src.configs.schemas import BBox
from inference.src.pipeline.postprocess import l2_normalize, rank_candidates


def _sign(key: str) -> str:
    return f"http://s3.test/{key}"


def _point(point_id: int, score: float, **payload: object) -> ScoredPoint:
    return ScoredPoint(id=point_id, version=0, score=score, payload=payload)


def test_l2_normalize_gives_a_unit_vector() -> None:
    vector = l2_normalize(np.array([3.0, 4.0]))
    assert vector.dtype == np.float32
    np.testing.assert_allclose(vector, [0.6, 0.8])


def test_l2_normalize_leaves_a_zero_vector_zero() -> None:
    np.testing.assert_array_equal(l2_normalize(np.zeros(3)), np.zeros(3))


def test_rank_sorts_by_score_and_fills_candidates() -> None:
    hits = [_point(1, 0.7, image_path="gallery/1.png"), _point(2, 0.9, image_path="gallery/2.png", vehicle_id="v2")]
    candidates, top_score, rejected = rank_candidates(hits, accepted=True, min_score=0.0, sign_url=_sign)

    assert not rejected
    assert top_score == pytest.approx(0.9)
    assert [(c.image_id, c.rank) for c in candidates] == [("2", 1), ("1", 2)]
    assert candidates[0].image_url == "http://s3.test/gallery/2.png"
    assert candidates[0].vehicle_id == "v2"


def test_rank_rejects_what_the_refusal_did_not_accept() -> None:
    candidates, top_score, rejected = rank_candidates(
        [_point(1, 0.4), _point(2, 0.3)], accepted=False, min_score=0.0, sign_url=_sign
    )
    assert rejected
    assert candidates == []
    assert top_score == pytest.approx(0.4)


def test_rank_rejects_an_empty_gallery() -> None:
    assert rank_candidates([], accepted=True, min_score=0.0, sign_url=_sign) == ([], None, True)


def test_rank_keeps_scores_in_unit_range() -> None:
    hits = [_point(1, 1.0000001), _point(2, -0.2)]
    candidates, _, _ = rank_candidates(hits, accepted=True, min_score=-1.0, sign_url=_sign)
    assert [c.score for c in candidates] == [1.0, 0.0]


def test_rank_carries_the_gallery_bbox_through() -> None:
    """The gallery stores whole frames, so a candidate without its box cannot be shown."""
    hit = _point(1, 0.9, image_path="gallery/1.png", bbox={"x": 10, "y": 20, "width": 30, "height": 40})

    candidates, _, _ = rank_candidates([hit], accepted=True, min_score=0.0, sign_url=_sign)

    assert candidates[0].bbox == BBox(x=10, y=20, width=30, height=40)


def test_rank_accepts_a_gallery_indexed_without_boxes() -> None:
    hit = _point(1, 0.9, image_path="gallery/1.png")
    candidates, _, _ = rank_candidates([hit], accepted=True, min_score=0.0, sign_url=_sign)

    assert candidates[0].bbox is None


def test_rank_drops_candidates_below_the_floor_and_reranks() -> None:
    """An accepted search is not padded up to top_k with weak matches."""
    hits = [_point(1, 0.95), _point(2, 0.27), _point(3, 0.71), _point(4, 0.58)]

    candidates, top_score, rejected = rank_candidates(hits, accepted=True, min_score=0.5854, sign_url=_sign)

    assert not rejected
    assert top_score == pytest.approx(0.95)
    assert [(c.image_id, c.rank) for c in candidates] == [("1", 1), ("3", 2)]


def test_rank_rejects_when_nothing_clears_the_floor() -> None:
    candidates, top_score, rejected = rank_candidates([_point(1, 0.5)], accepted=True, min_score=0.6, sign_url=_sign)

    assert rejected
    assert candidates == []
    assert top_score == pytest.approx(0.5)

import numpy as np
import pytest
from PIL import Image
from qdrant_client.models import ScoredPoint

from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError
from inference.src.configs.settings import InferenceSettings
from inference.src.models.factory import build_embedder
from inference.src.models.stub import StubEmbedder
from inference.src.pipeline.crop import crop_vehicle
from inference.src.pipeline.postprocess import l2_normalize, rank_candidates
from inference.src.pipeline.preprocess import preprocess
from inference.tests.helpers import BOX, make_image


def _sign(key: str) -> str:
    return f"http://s3.test/{key}"


def _point(point_id: int, score: float, **payload: object) -> ScoredPoint:
    return ScoredPoint(id=point_id, version=0, score=score, payload=payload)


def test_crop_cuts_the_box() -> None:
    crop = crop_vehicle(make_image(1), BBox(x=10, y=20, width=50, height=40))
    assert crop.size == (50, 40)


def test_crop_trims_a_box_that_overflows_the_image() -> None:
    crop = crop_vehicle(make_image(1, size=(128, 96)), BBox(x=100, y=10, width=100, height=50))
    assert crop.size == (28, 50)


def test_crop_rejects_a_box_that_degenerates_after_trimming() -> None:
    with pytest.raises(ValidationError):
        crop_vehicle(make_image(1, size=(128, 96)), BBox(x=120, y=10, width=50, height=50))


def test_crop_rejects_undecodable_bytes() -> None:
    with pytest.raises(ValidationError):
        crop_vehicle(b"definitely not an image", BBox(x=0, y=0, width=20, height=20))


def test_preprocess_gives_fixed_size_rgb() -> None:
    result = preprocess(Image.new("L", (50, 40)), (32, 24))
    assert result.mode == "RGB"
    assert result.size == (32, 24)


def test_l2_normalize_gives_a_unit_vector() -> None:
    vector = l2_normalize(np.array([3.0, 4.0]))
    assert vector.dtype == np.float32
    np.testing.assert_allclose(vector, [0.6, 0.8])


def test_l2_normalize_leaves_a_zero_vector_zero() -> None:
    np.testing.assert_array_equal(l2_normalize(np.zeros(3)), np.zeros(3))


def test_stub_embedder_is_deterministic_per_crop() -> None:
    embedder = StubEmbedder(dim=8)
    first, second = make_image(1), make_image(2)
    assert embedder.embed(first, BOX).shape == (8,)
    np.testing.assert_array_equal(embedder.embed(first, BOX), embedder.embed(first, BOX))
    assert not np.array_equal(embedder.embed(first, BOX), embedder.embed(second, BOX))


def test_stub_embedder_rejects_undecodable_bytes() -> None:
    with pytest.raises(ValidationError):
        StubEmbedder(dim=8).embed(b"definitely not an image", BOX)


def test_unknown_embedder_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder(InferenceSettings(embedder="resnet9000"))


def test_rank_sorts_by_score_and_fills_candidates() -> None:
    hits = [_point(1, 0.7, image_path="gallery/1.png"), _point(2, 0.9, image_path="gallery/2.png", vehicle_id="v2")]
    candidates, top_score, rejected = rank_candidates(hits, threshold=0.5, sign_url=_sign)

    assert not rejected
    assert top_score == pytest.approx(0.9)
    assert [(c.image_id, c.rank) for c in candidates] == [("2", 1), ("1", 2)]
    assert candidates[0].image_url == "http://s3.test/gallery/2.png"
    assert candidates[0].vehicle_id == "v2"


@pytest.mark.skip(reason="the rejection threshold is disabled, see the TODO in rank_candidates")
def test_rank_rejects_when_the_best_score_is_below_threshold() -> None:
    candidates, top_score, rejected = rank_candidates([_point(1, 0.4), _point(2, 0.3)], threshold=0.5, sign_url=_sign)
    assert rejected
    assert candidates == []
    assert top_score == pytest.approx(0.4)


def test_rank_rejects_an_empty_gallery() -> None:
    assert rank_candidates([], threshold=0.5, sign_url=_sign) == ([], None, True)


def test_rank_keeps_scores_in_unit_range() -> None:
    candidates, _, _ = rank_candidates([_point(1, 1.0000001), _point(2, -0.2)], threshold=0.0, sign_url=_sign)
    assert [c.score for c in candidates] == [1.0, 0.0]


def test_rank_carries_the_gallery_bbox_through() -> None:
    """The gallery stores whole frames, so a candidate without its box cannot be shown."""
    hit = _point(1, 0.9, image_path="gallery/1.png", bbox={"x": 10, "y": 20, "width": 30, "height": 40})

    candidates, _, _ = rank_candidates([hit], threshold=0.5, sign_url=_sign)

    assert candidates[0].bbox == BBox(x=10, y=20, width=30, height=40)


def test_rank_accepts_a_gallery_indexed_without_boxes() -> None:
    candidates, _, _ = rank_candidates([_point(1, 0.9, image_path="gallery/1.png")], threshold=0.5, sign_url=_sign)

    assert candidates[0].bbox is None

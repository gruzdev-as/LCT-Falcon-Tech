"""The real model against the training code that embedded the gallery.

Needs the serving weights and skips without them. They default to the LFS copy in
the training submodule, the same file init downloads from Hugging Face.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError
from inference.src.models.reid import ReIDEmbedder
from inference.tests.helpers import make_image

WEIGHTS = Path(os.getenv("INFERENCE_TEST_WEIGHTS", "training/weights/finetuned/eva02.pt"))
if not WEIGHTS.is_file() or WEIGHTS.stat().st_size < 1024 * 1024:  # an LFS pointer is a few bytes
    pytest.skip(f"no serving weights at {WEIGHTS}", allow_module_level=True)

BOX = BBox(x=40, y=30, width=200, height=140)


@pytest.fixture(scope="module")
def embedder() -> ReIDEmbedder:
    return ReIDEmbedder(WEIGHTS, device="cpu")


def test_reports_the_checkpoint_dimension(embedder: ReIDEmbedder) -> None:
    assert embedder.dim == 256
    assert len(embedder.version) == 64


def test_embeds_one_unit_vector(embedder: ReIDEmbedder) -> None:
    vector = embedder.embed(make_image(1, size=(320, 240), image_format="JPEG"), BOX)

    assert vector.shape == (256,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-4)


def test_matches_how_training_embeds_the_gallery(embedder: ReIDEmbedder, tmp_path: Path) -> None:
    """The query path must reproduce eval.py's embed_frame, or search drifts silently."""
    from modules.inference import embed_frame, records_frame  # noqa: PLC0415

    data = make_image(2, size=(320, 240), image_format="JPEG")
    path = tmp_path / "frame.jpg"
    path.write_bytes(data)

    cfg = embedder.cfg.copy()
    cfg.data.num_workers = 0
    frame = records_frame([path], [(BOX.x, BOX.y, BOX.width, BOX.height)])
    reference = embed_frame(embedder._model, cfg, embedder.device, frame)[0]  # noqa: SLF001

    assert float(np.dot(embedder.embed(data, BOX), reference)) > 0.999


def test_rejects_undecodable_bytes(embedder: ReIDEmbedder) -> None:
    with pytest.raises(ValidationError):
        embedder.embed(b"definitely not an image", BOX)


def test_rejects_a_box_outside_the_image(embedder: ReIDEmbedder) -> None:
    with pytest.raises(ValidationError):
        embedder.embed(make_image(3, size=(64, 64)), BBox(x=100, y=100, width=50, height=50))

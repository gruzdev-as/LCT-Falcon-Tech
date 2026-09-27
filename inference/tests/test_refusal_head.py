"""The CatBoost refusal against training's own decision on the whole gallery.

Needs the head and skips without it. It defaults to the LFS copy in the training
submodule, the same file init downloads.
"""

import os
import uuid
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from refusal import refusal_accept

from inference.src.configs.constants import REFUSAL_CONFIG
from inference.src.configs.refusal import load_refusal_preset
from inference.src.models.refusal_head import HeadRefusal

HEAD = Path(os.getenv("INFERENCE_TEST_REFUSAL_HEAD", "training/weights/finetuned/eva02_catboost.cbm"))
if not HEAD.is_file() or HEAD.stat().st_size < 10_000:  # an LFS pointer is a few bytes
    pytest.skip(f"no refusal head at {HEAD}", allow_module_level=True)

DIM = 256
PRESET = load_refusal_preset(REFUSAL_CONFIG)


def _unit(rows: np.ndarray) -> np.ndarray:
    return (rows / np.linalg.norm(rows, axis=-1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def head() -> HeadRefusal:
    return HeadRefusal(PRESET, HEAD)


@pytest.fixture
async def gallery() -> tuple[AsyncQdrantClient, np.ndarray]:
    """Random vehicles plus a few close variants of the first, the way one car recurs."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((60, DIM))
    variants = base[0] + 0.35 * rng.standard_normal((4, DIM))
    vectors = _unit(np.concatenate([base, variants]))

    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection("gallery", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    points = [PointStruct(id=str(uuid.uuid4()), vector=row.tolist()) for row in vectors]
    await client.upsert("gallery", points, wait=True)
    yield client, vectors
    await client.close()


async def _decide(head: HeadRefusal, client: AsyncQdrantClient, query: np.ndarray) -> bool:
    response = await client.query_points("gallery", query=query.tolist(), limit=head.neighbours, with_vectors=True)
    return head.accept(query, response.points)


async def test_matches_training_on_the_whole_gallery(head: HeadRefusal, gallery) -> None:  # noqa: ANN001
    """Top-k from Qdrant must give the answer training gives with every gallery vector."""
    client, vectors = gallery
    # Copies of gallery vehicles under growing noise: from near-duplicates, which the
    # head accepts, to strangers, which it refuses — the boundary is where it matters.
    noise = np.linspace(0.02, 0.2, 16)[:, None]
    queries = _unit(vectors[:16] + noise * np.random.default_rng(1).standard_normal((16, DIM)))

    expected = refusal_accept(
        PRESET.kind,
        queries,
        vectors,
        cosine_threshold=PRESET.cosine_threshold,
        model=head._model,  # noqa: SLF001
        model_threshold=PRESET.model_threshold,
        k=PRESET.k,
        with_embeddings=PRESET.with_embeddings,
    )
    decided = [await _decide(head, client, query) for query in queries]

    assert expected.any()
    assert not expected.all()
    assert decided == expected.tolist()


async def test_rejects_a_vehicle_that_is_not_in_the_gallery(head: HeadRefusal, gallery) -> None:  # noqa: ANN001
    client, _ = gallery
    stranger = _unit(np.random.default_rng(7).standard_normal(DIM))
    assert not await _decide(head, client, stranger)


def test_a_missing_head_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        HeadRefusal(PRESET, tmp_path / "missing.cbm")

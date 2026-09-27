import pytest
from qdrant_client import AsyncQdrantClient

from common.src.configs.schemas import TaskStatus
from inference.src.models.refusal import CosineRefusal
from inference.src.models.stub import StubEmbedder
from inference.src.processor import Processor
from inference.tests.helpers import GalleryAdd, make_image, make_task


@pytest.fixture
def processor(embedder: StubEmbedder, qdrant: AsyncQdrantClient, objects: dict[str, bytes]) -> Processor:
    return Processor(embedder=embedder, refusal=CosineRefusal(0.5))


async def test_finds_the_same_vehicle(processor: Processor, objects: dict[str, bytes], gallery_add: GalleryAdd) -> None:
    image = make_image(1)
    match_id = await gallery_add(image, vehicle_id="car-1")
    await gallery_add(make_image(2))
    objects["queries/q.png"] = image

    result = await processor.process(make_task())

    assert result.status == TaskStatus.DONE
    assert not result.rejected
    assert result.top_score == pytest.approx(1.0, abs=1e-5)
    assert result.candidates[0].image_id == match_id
    assert result.candidates[0].vehicle_id == "car-1"
    assert result.candidates[0].image_url.startswith("http://s3.test/gallery/")
    assert result.model_name == "stub"


async def test_rejects_when_nothing_is_close(
    processor: Processor, objects: dict[str, bytes], gallery_add: GalleryAdd
) -> None:
    await gallery_add(make_image(2))
    objects["queries/q.png"] = make_image(1)

    result = await processor.process(make_task())

    assert result.status == TaskStatus.DONE
    assert result.rejected
    assert result.candidates == []


async def test_rejects_on_an_empty_gallery(processor: Processor, objects: dict[str, bytes]) -> None:
    objects["queries/q.png"] = make_image(1)
    result = await processor.process(make_task())
    assert result.rejected
    assert result.top_score is None


async def test_returns_at_most_top_k(processor: Processor, objects: dict[str, bytes], gallery_add: GalleryAdd) -> None:
    image = make_image(1)
    for _ in range(4):
        await gallery_add(image)
    objects["queries/q.png"] = image

    result = await processor.process(make_task(top_k=2))

    assert [c.rank for c in result.candidates] == [1, 2]


async def test_fails_an_undecodable_image(processor: Processor, objects: dict[str, bytes]) -> None:
    objects["queries/q.png"] = b"not an image"
    result = await processor.process(make_task())
    assert result.status == TaskStatus.FAILED
    assert result.error == "stored image cannot be decoded"


async def test_fails_a_missing_image(processor: Processor) -> None:
    result = await processor.process(make_task("queries/missing.png"))
    assert result.status == TaskStatus.FAILED
    assert "does not exist" in result.error


async def test_fails_a_box_that_degenerates_on_the_image(processor: Processor, objects: dict[str, bytes]) -> None:
    objects["queries/q.png"] = make_image(1, size=(20, 20))  # BOX starts at (10, 10): 10px left
    result = await processor.process(make_task())
    assert result.status == TaskStatus.FAILED


class _RecordingRefusal:
    """Accepts everything and remembers what the processor showed it."""

    name = "recording"
    neighbours = 3
    needs_vectors = True
    min_score = 0.0

    def __init__(self) -> None:
        self.seen: list[list] = []

    def accept(self, query, hits) -> bool:  # noqa: ANN001
        self.seen.append([hit.vector for hit in hits])
        return True


async def test_the_refusal_sees_its_neighbours_with_vectors(
    embedder: StubEmbedder, objects: dict[str, bytes], gallery_add: GalleryAdd
) -> None:
    """The head needs more neighbours than a top-1 caller asks for, and their vectors."""
    for seed in range(5):
        await gallery_add(make_image(seed))
    objects["queries/q.png"] = make_image(0)
    refusal = _RecordingRefusal()

    result = await Processor(embedder=embedder, refusal=refusal).process(make_task(top_k=1))

    assert len(result.candidates) == 1
    assert len(refusal.seen[0]) == 3
    assert all(len(vector) == embedder.dim for vector in refusal.seen[0])


async def test_a_refused_query_answers_rejected_with_its_top_score(
    embedder: StubEmbedder, objects: dict[str, bytes], gallery_add: GalleryAdd
) -> None:
    image = make_image(1)
    await gallery_add(image)
    objects["queries/q.png"] = image

    result = await Processor(embedder=embedder, refusal=CosineRefusal(1.01)).process(make_task())

    assert result.rejected
    assert result.candidates == []
    assert result.top_score == pytest.approx(1.0, abs=1e-5)


async def test_an_accepted_search_returns_only_strong_candidates(
    processor: Processor, objects: dict[str, bytes], gallery_add: GalleryAdd
) -> None:
    image = make_image(1)
    match_id = await gallery_add(image)
    for seed in range(2, 6):
        await gallery_add(make_image(seed))
    objects["queries/q.png"] = image

    result = await processor.process(make_task(top_k=5))

    assert not result.rejected
    assert [c.image_id for c in result.candidates] == [match_id]

import fakeredis
import pytest
from qdrant_client import AsyncQdrantClient

from common.src.configs.constants import GALLERY_COLLECTION, GALLERY_FAILED_KEY
from common.src.qdrant.gallery import point_id
from inference.src.gallery import ForeignModelError, GalleryIndexer
from inference.src.pipeline.embed import embed_vector
from inference.tests.helpers import BOX, FakeEmbedder, make_gallery_task, make_image


@pytest.fixture
def indexer(
    embedder: FakeEmbedder, qdrant: AsyncQdrantClient, redis: fakeredis.FakeAsyncRedis, objects: dict[str, bytes]
) -> GalleryIndexer:
    return GalleryIndexer(embedder)


async def test_indexes_the_image_the_way_a_search_embeds_it(
    indexer: GalleryIndexer, embedder: FakeEmbedder, objects: dict[str, bytes], qdrant: AsyncQdrantClient
) -> None:
    task = make_gallery_task()
    objects[task.image_path] = make_image(1)

    assert await indexer.index(task)

    point = (await qdrant.retrieve(GALLERY_COLLECTION, [point_id(task.image_id)], with_vectors=True))[0]
    assert point.payload["image_id"] == task.image_id
    assert point.payload["image_path"] == task.image_path
    assert point.payload["bbox"] == BOX.model_dump()
    assert point.payload["model_version"] == embedder.version
    assert point.vector == pytest.approx(embed_vector(embedder, make_image(1), BOX).tolist(), abs=1e-6)


async def test_a_broken_image_is_skipped_not_retried(
    indexer: GalleryIndexer, objects: dict[str, bytes], redis: fakeredis.FakeAsyncRedis, qdrant: AsyncQdrantClient
) -> None:
    task = make_gallery_task()
    objects[task.image_path] = b"not an image"

    assert not await indexer.index(task)

    assert await redis.smembers(GALLERY_FAILED_KEY) == {task.image_id}
    assert (await qdrant.count(GALLERY_COLLECTION)).count == 0


async def test_a_missing_original_is_skipped(indexer: GalleryIndexer, redis: fakeredis.FakeAsyncRedis) -> None:
    task = make_gallery_task()

    assert not await indexer.index(task)
    assert await redis.sismember(GALLERY_FAILED_KEY, task.image_id)


async def test_refuses_a_task_for_other_weights(indexer: GalleryIndexer, objects: dict[str, bytes]) -> None:
    """A vector from other weights would fit the collection and silently rank nonsense."""
    task = make_gallery_task(model_version="f" * 64)
    objects[task.image_path] = make_image(1)

    with pytest.raises(ForeignModelError):
        await indexer.index(task)


async def test_recreates_a_collection_dropped_under_it(
    indexer: GalleryIndexer, objects: dict[str, bytes], qdrant: AsyncQdrantClient
) -> None:
    """A forced init drops the collection while the workers keep running."""
    await qdrant.delete_collection(GALLERY_COLLECTION)
    task = make_gallery_task()
    objects[task.image_path] = make_image(1)

    assert await indexer.index(task)
    assert (await qdrant.count(GALLERY_COLLECTION)).count == 1

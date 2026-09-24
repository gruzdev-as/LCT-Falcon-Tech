import pytest
from qdrant_client import AsyncQdrantClient

from common.src.configs.constants import GALLERY_COLLECTION, GALLERY_PREFIX
from common.src.exceptions import StorageError
from common.src.qdrant import client as qdrant
from init.src.bundle import Bundle
from init.src.steps.images import object_key, upload_gallery
from init.src.steps.vectors import index_gallery, point_id


async def test_uploads_every_gallery_image(bundle: Bundle, objects: dict[str, bytes]) -> None:
    uploaded = await upload_gallery(bundle, concurrency=4)

    assert uploaded == bundle.count
    assert len(objects) == bundle.count
    assert all(key.startswith(f"{GALLERY_PREFIX}/") for key in objects)


async def test_a_second_upload_writes_nothing(bundle: Bundle, objects: dict[str, bytes]) -> None:
    await upload_gallery(bundle, concurrency=4)

    assert await upload_gallery(bundle, concurrency=4) == 0
    assert len(objects) == bundle.count


async def test_upload_fills_only_the_gaps(bundle: Bundle, objects: dict[str, bytes]) -> None:
    await upload_gallery(bundle, concurrency=4)
    del objects[object_key(bundle.entries[0])]

    assert await upload_gallery(bundle, concurrency=4) == 1


async def test_force_reuploads_everything(bundle: Bundle, objects: dict[str, bytes]) -> None:
    await upload_gallery(bundle, concurrency=4)

    assert await upload_gallery(bundle, concurrency=4, force=True) == bundle.count


async def test_indexes_the_bundle(bundle: Bundle, qdrant_client: AsyncQdrantClient) -> None:
    indexed = await index_gallery(bundle, batch_size=2)

    assert indexed == bundle.count
    assert await qdrant.count_points(GALLERY_COLLECTION) == bundle.count


async def test_payload_carries_what_the_search_result_reads(bundle: Bundle, qdrant_client: AsyncQdrantClient) -> None:
    await index_gallery(bundle, batch_size=8)

    entry = bundle.entries[0]
    point = (await qdrant_client.retrieve(GALLERY_COLLECTION, [point_id(entry.image_id)], with_payload=True))[0]

    assert point.payload["image_id"] == entry.image_id
    assert point.payload["image_path"] == object_key(entry)
    # Written as text because Candidate types these as `str | None`.
    assert isinstance(point.payload["vehicle_id"], str)
    assert isinstance(point.payload["camera_id"], str)


async def test_reindexing_the_same_gallery_is_a_no_op(bundle: Bundle, qdrant_client: AsyncQdrantClient) -> None:
    await index_gallery(bundle, batch_size=8)

    assert await index_gallery(bundle, batch_size=8) == 0


async def test_a_wiped_collection_is_rebuilt(bundle: Bundle, qdrant_client: AsyncQdrantClient) -> None:
    """Deleting data/qdrant must repair itself on the next run, without --force."""
    await index_gallery(bundle, batch_size=8)
    await qdrant_client.delete_collection(GALLERY_COLLECTION)

    assert await index_gallery(bundle, batch_size=8) == bundle.count


async def test_force_drops_the_previous_vectors(bundle: Bundle, qdrant_client: AsyncQdrantClient) -> None:
    """Replacing the artifact: topping up would leave an unsearchable mix of models."""
    await index_gallery(bundle, batch_size=8)
    smaller = _variant(bundle, version="next", entries=bundle.entries[:2], embeddings=bundle.embeddings[:2])

    await index_gallery(smaller, batch_size=8, force=True)

    assert await qdrant.count_points(GALLERY_COLLECTION) == 2


async def test_force_resizes_the_collection_for_a_new_dimension(
    bundle: Bundle, qdrant_client: AsyncQdrantClient
) -> None:
    await index_gallery(bundle, batch_size=8)
    narrower = _variant(bundle, embedding_dim=4, embeddings=bundle.embeddings[:, :4])

    await index_gallery(narrower, batch_size=8, force=True)

    vectors = (await qdrant_client.get_collection(GALLERY_COLLECTION)).config.params.vectors
    assert vectors.size == 4


async def test_a_dimension_mismatch_is_loud(bundle: Bundle, qdrant_client: AsyncQdrantClient) -> None:
    """Never silently write into a collection sized for another model."""
    await qdrant.ensure_collection(GALLERY_COLLECTION, bundle.embedding_dim + 1)

    with pytest.raises(StorageError, match="dim"):
        await index_gallery(bundle, batch_size=8)


def _variant(bundle: Bundle, **overrides) -> Bundle:
    fields = {
        "version": bundle.version,
        "model_name": bundle.model_name,
        "embedding_dim": bundle.embedding_dim,
        "distance": bundle.distance,
        "entries": bundle.entries,
        "embeddings": bundle.embeddings,
        "images_dir": bundle.images_dir,
    }
    return Bundle(**(fields | overrides))


def test_point_ids_are_deterministic() -> None:
    assert point_id("img-1") == point_id("img-1")
    assert point_id("img-1") != point_id("img-2")

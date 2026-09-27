import fakeredis
import pytest
from qdrant_client import AsyncQdrantClient

from common.src.configs.constants import (
    GALLERY_COLLECTION,
    GALLERY_FAILED_KEY,
    GALLERY_STATE_KEY,
    GALLERY_STREAM,
    INFERENCE_GROUP,
    STREAM_PAYLOAD_FIELD,
)
from common.src.configs.schemas import GalleryTask
from common.src.qdrant import client as qdrant
from common.src.qdrant.gallery import gallery_point, point_id
from common.src.redis.client import stream_backlog
from init.src.gallery import Gallery, GalleryEntry
from init.src.steps.enqueue import schedule_gallery
from init.src.steps.images import object_key

WEIGHTS = "a" * 64
OTHER_WEIGHTS = "b" * 64


async def _queued(redis: fakeredis.FakeAsyncRedis) -> list[GalleryTask]:
    entries = await redis.xrange(GALLERY_STREAM)
    return [GalleryTask.model_validate_json(fields[STREAM_PAYLOAD_FIELD]) for _, fields in entries]


async def _index(entries: list[GalleryEntry], version: str) -> None:
    """Stand in for a worker that already embedded these images."""
    await qdrant.ensure_collection(GALLERY_COLLECTION, 4)
    points = [
        gallery_point(
            image_id=entry.image_id,
            image_path=object_key(entry),
            bbox=entry.bbox,
            vehicle_id=entry.vehicle_id,
            camera_id=entry.camera_id,
            model_version=version,
            vector=[1.0, 0.0, 0.0, 0.0],
        )
        for entry in entries
    ]
    await qdrant.upsert_points(GALLERY_COLLECTION, points)


@pytest.fixture(autouse=True)
def _stores(redis: fakeredis.FakeAsyncRedis, qdrant_client: AsyncQdrantClient) -> None:
    """Every test here needs both stores."""


async def test_queues_every_image_of_a_new_gallery(gallery: Gallery, redis: fakeredis.FakeAsyncRedis) -> None:
    schedule = await schedule_gallery(gallery, WEIGHTS)

    tasks = await _queued(redis)
    assert schedule.queued == gallery.count
    assert [task.image_id for task in tasks] == [entry.image_id for entry in gallery.entries]
    first, entry = tasks[0], gallery.entries[0]
    assert first.image_path == object_key(entry)
    assert first.bbox == entry.bbox
    assert first.model_version == WEIGHTS


async def test_records_what_a_finished_gallery_looks_like(gallery: Gallery, redis: fakeredis.FakeAsyncRedis) -> None:
    await redis.sadd(GALLERY_FAILED_KEY, "left-from-last-run")

    await schedule_gallery(gallery, WEIGHTS)

    assert await redis.hgetall(GALLERY_STATE_KEY) == {"model_version": WEIGHTS, "total": str(gallery.count)}
    assert not await redis.exists(GALLERY_FAILED_KEY)


async def test_an_indexed_gallery_queues_nothing(gallery: Gallery, redis: fakeredis.FakeAsyncRedis) -> None:
    await _index(gallery.entries, WEIGHTS)

    schedule = await schedule_gallery(gallery, WEIGHTS)

    assert schedule.queued == 0
    assert schedule.already_indexed == gallery.count
    assert await _queued(redis) == []


async def test_queues_only_the_gaps(gallery: Gallery, redis: fakeredis.FakeAsyncRedis) -> None:
    await _index(gallery.entries[1:], WEIGHTS)

    await schedule_gallery(gallery, WEIGHTS)

    assert [task.image_id for task in await _queued(redis)] == [gallery.entries[0].image_id]


async def test_does_not_queue_twice_while_the_workers_catch_up(
    gallery: Gallery, redis: fakeredis.FakeAsyncRedis
) -> None:
    await schedule_gallery(gallery, WEIGHTS)

    schedule = await schedule_gallery(gallery, WEIGHTS)

    assert schedule.queued == 0
    assert len(await _queued(redis)) == gallery.count


async def test_replaces_points_from_other_weights(gallery: Gallery, redis: fakeredis.FakeAsyncRedis) -> None:
    """A vector from other weights has the right size and silently ranks nonsense."""
    await _index(gallery.entries, OTHER_WEIGHTS)

    schedule = await schedule_gallery(gallery, WEIGHTS)

    assert schedule.removed == gallery.count
    assert schedule.queued == gallery.count
    assert await qdrant.count_points(GALLERY_COLLECTION) == 0


async def test_drops_images_that_left_the_manifest(gallery: Gallery) -> None:
    await _index(gallery.entries, WEIGHTS)
    shrunk = Gallery(version="shrunk", entries=gallery.entries[:-1], images_dir=gallery.images_dir)

    schedule = await schedule_gallery(shrunk, WEIGHTS)

    assert schedule.removed == 1
    assert schedule.queued == 0
    gone = point_id(gallery.entries[-1].image_id)
    assert await qdrant.get_qdrant().retrieve(GALLERY_COLLECTION, [gone]) == []


async def test_force_embeds_everything_again(gallery: Gallery, redis: fakeredis.FakeAsyncRedis) -> None:
    await _index(gallery.entries, WEIGHTS)
    await schedule_gallery(gallery, WEIGHTS)

    schedule = await schedule_gallery(gallery, WEIGHTS, force=True)

    assert schedule.queued == gallery.count
    assert not await qdrant.get_qdrant().collection_exists(GALLERY_COLLECTION)
    assert len(await _queued(redis)) == gallery.count


async def test_counts_what_the_workers_hold_as_still_queued(gallery: Gallery, redis: fakeredis.FakeAsyncRedis) -> None:
    """Delivered-but-unacked and never-delivered tasks both mean the last run is not done."""
    await schedule_gallery(gallery, WEIGHTS)
    await redis.xgroup_create(GALLERY_STREAM, INFERENCE_GROUP, id="0")
    done = await redis.xreadgroup(INFERENCE_GROUP, "worker", {GALLERY_STREAM: ">"}, count=1)
    await redis.xack(GALLERY_STREAM, INFERENCE_GROUP, done[0][1][0][0])
    await redis.xreadgroup(INFERENCE_GROUP, "worker", {GALLERY_STREAM: ">"}, count=1)

    assert await stream_backlog(GALLERY_STREAM, INFERENCE_GROUP) == gallery.count - 1
    assert (await schedule_gallery(gallery, WEIGHTS)).queued == 0

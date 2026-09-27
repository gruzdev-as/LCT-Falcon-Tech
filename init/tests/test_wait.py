from pathlib import Path

import fakeredis
import pytest
from qdrant_client import AsyncQdrantClient

from common.src.configs.constants import GALLERY_COLLECTION, GALLERY_FAILED_KEY, GALLERY_STATE_KEY
from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError
from common.src.qdrant import client as qdrant
from common.src.qdrant.gallery import gallery_point
from init.src.configs.settings import InitSettings
from init.src.wait import read_progress, wait_for_gallery

WEIGHTS = "a" * 64
BOX = BBox(x=0, y=0, width=20, height=20)


async def _index(*image_ids: str, version: str = WEIGHTS) -> None:
    await qdrant.ensure_collection(GALLERY_COLLECTION, 4)
    points = [
        gallery_point(
            image_id=image_id,
            image_path=f"gallery/{image_id}.png",
            bbox=BOX,
            vehicle_id=None,
            camera_id=None,
            model_version=version,
            vector=[1.0, 0.0, 0.0, 0.0],
        )
        for image_id in image_ids
    ]
    await qdrant.upsert_points(GALLERY_COLLECTION, points)


class _Clock:
    """Time that moves only when the wait sleeps, so a test never really waits."""

    def __init__(self) -> None:
        self.now = 0.0
        self.on_sleep: list = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep:
            await self.on_sleep.pop(0)()


@pytest.fixture
async def expecting_three(redis: fakeredis.FakeAsyncRedis, qdrant_client: AsyncQdrantClient) -> None:
    await redis.hset(GALLERY_STATE_KEY, mapping={"model_version": WEIGHTS, "total": 3})


def _settings(tmp_path: Path, **overrides: object) -> InitSettings:
    return InitSettings(gallery_dir=tmp_path, **({"wait_poll_s": 1.0} | overrides))


async def test_counts_only_points_with_the_expected_weights(expecting_three: None) -> None:
    await _index("a", "b")
    await _index("c", version="b" * 64)

    progress = await read_progress()

    assert (progress.indexed, progress.failed, progress.done) == (2, 0, False)


async def test_waits_until_the_workers_are_done(expecting_three: None, tmp_path: Path) -> None:
    clock = _Clock()
    clock.on_sleep = [lambda: _index("a"), lambda: _index("b", "c")]

    progress = await wait_for_gallery(_settings(tmp_path), clock=clock, sleep=clock.sleep)

    assert progress.indexed == 3
    assert clock.now == 2.0


async def test_broken_images_do_not_hold_the_gallery_back(
    expecting_three: None, redis: fakeredis.FakeAsyncRedis, tmp_path: Path
) -> None:
    await _index("a", "b")
    await redis.sadd(GALLERY_FAILED_KEY, "c")

    progress = await wait_for_gallery(_settings(tmp_path))

    assert (progress.indexed, progress.failed, progress.done) == (2, 1, True)


async def test_nothing_indexed_is_an_error(
    expecting_three: None, redis: fakeredis.FakeAsyncRedis, tmp_path: Path
) -> None:
    """Every image failing is a misconfigured worker, not a few bad files."""
    await redis.sadd(GALLERY_FAILED_KEY, "a", "b", "c")

    with pytest.raises(ValidationError, match="indexed no gallery image"):
        await wait_for_gallery(_settings(tmp_path))


async def test_gives_up_after_the_timeout(expecting_three: None, tmp_path: Path) -> None:
    clock = _Clock()

    with pytest.raises(TimeoutError):
        await wait_for_gallery(_settings(tmp_path, wait_timeout_s=10.0), clock=clock, sleep=clock.sleep)


async def test_without_init_there_is_nothing_to_wait_for(
    redis: fakeredis.FakeAsyncRedis, qdrant_client: AsyncQdrantClient
) -> None:
    with pytest.raises(ValidationError, match="init has not queued"):
        await read_progress()

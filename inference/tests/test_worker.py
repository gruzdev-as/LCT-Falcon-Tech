from pathlib import Path

import fakeredis
import pytest
from qdrant_client import AsyncQdrantClient

from common.src.configs.constants import INFERENCE_GROUP, RESULT_KEY, TASK_KEY, TASK_STREAM
from common.src.configs.schemas import EmbeddingTask, SearchResult, TaskStatus
from common.src.qdrant import client as qdrant_module
from common.src.redis.client import read_one
from inference.src.configs.constants import POISON_ERROR
from inference.src.configs.settings import InferenceSettings
from inference.src.models.refusal import CosineRefusal
from inference.src.models.stub import StubEmbedder
from inference.src.processor import Processor
from inference.src.worker import InferenceWorker
from inference.tests.helpers import Submit, make_image


def _settings(tmp_path: Path, **overrides: object) -> InferenceSettings:
    base = {
        "block_ms": 10,
        "claim_idle_ms": 0,
        "claim_interval_s": 0,
        "max_deliveries": 3,
        "heartbeat_path": tmp_path / "alive",
    }
    return InferenceSettings(**(base | overrides))


def _worker(embedder: StubEmbedder, settings: InferenceSettings, consumer: str = "live") -> InferenceWorker:
    return InferenceWorker(Processor(embedder, CosineRefusal(0.5)), settings, consumer=consumer)


@pytest.fixture
async def worker(
    embedder: StubEmbedder, qdrant: AsyncQdrantClient, redis: fakeredis.FakeAsyncRedis, tmp_path: Path
) -> InferenceWorker:
    instance = _worker(embedder, _settings(tmp_path))
    await instance.start()
    return instance


async def _result(redis: fakeredis.FakeAsyncRedis, task: EmbeddingTask) -> SearchResult | None:
    raw = await redis.get(RESULT_KEY.format(task_id=task.task_id))
    return None if raw is None else SearchResult.model_validate_json(raw)


async def _pending(redis: fakeredis.FakeAsyncRedis) -> int:
    return (await redis.xpending(TASK_STREAM, INFERENCE_GROUP))["pending"]


async def test_answers_a_task_and_retires_it(
    worker: InferenceWorker, submit: Submit, redis: fakeredis.FakeAsyncRedis
) -> None:
    task = await submit(make_image(1))

    assert await worker.run_once()

    result = await _result(redis, task)
    assert result is not None
    assert result.status == TaskStatus.DONE
    assert result.rejected  # the gallery is empty
    assert not await redis.exists(TASK_KEY.format(task_id=task.task_id))
    assert await _pending(redis) == 0


async def test_failed_task_is_still_answered_and_acked(
    worker: InferenceWorker, submit: Submit, redis: fakeredis.FakeAsyncRedis
) -> None:
    task = await submit(data=None)  # the original never reached storage

    await worker.run_once()

    result = await _result(redis, task)
    assert result.status == TaskStatus.FAILED
    assert result.error
    assert await _pending(redis) == 0


async def test_skips_a_task_nobody_waits_for(
    worker: InferenceWorker, submit: Submit, redis: fakeredis.FakeAsyncRedis
) -> None:
    task = await submit(make_image(1), live=False)

    assert not await worker.run_once()

    assert await _result(redis, task) is None
    assert await _pending(redis) == 0


async def test_infrastructure_failure_leaves_the_task_pending(
    worker: InferenceWorker, submit: Submit, redis: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def qdrant_down(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError

    monkeypatch.setattr(qdrant_module, "search", qdrant_down)
    task = await submit(make_image(1))

    with pytest.raises(ConnectionError):
        await worker.run_once()

    assert await _result(redis, task) is None
    assert await redis.exists(TASK_KEY.format(task_id=task.task_id))
    assert await _pending(redis) == 1


async def test_picks_up_the_task_of_a_dead_replica(
    worker: InferenceWorker, submit: Submit, redis: fakeredis.FakeAsyncRedis
) -> None:
    task = await submit(make_image(1))
    await read_one(TASK_STREAM, INFERENCE_GROUP, "dead", EmbeddingTask, block_ms=10)  # read, then crashed

    assert await worker.run_once()

    assert (await _result(redis, task)).status == TaskStatus.DONE
    assert await _pending(redis) == 0


async def test_fails_a_task_that_keeps_killing_workers(
    embedder: StubEmbedder,
    qdrant: AsyncQdrantClient,
    submit: Submit,
    redis: fakeredis.FakeAsyncRedis,
    tmp_path: Path,
) -> None:
    worker = _worker(embedder, _settings(tmp_path, max_deliveries=1))
    await worker.start()
    task = await submit(make_image(1))
    await read_one(TASK_STREAM, INFERENCE_GROUP, "dead", EmbeddingTask, block_ms=10)

    assert not await worker.run_once()

    result = await _result(redis, task)
    assert result.status == TaskStatus.FAILED
    assert result.error == POISON_ERROR
    assert await _pending(redis) == 0


async def test_scaled_out_workers_split_the_stream(
    embedder: StubEmbedder,
    qdrant: AsyncQdrantClient,
    submit: Submit,
    redis: fakeredis.FakeAsyncRedis,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first, second = _worker(embedder, settings, "a"), _worker(embedder, settings, "b")
    await first.start()
    await second.start()
    tasks = [await submit(make_image(seed)) for seed in range(2)]

    assert await first.run_once()
    assert await second.run_once()

    assert [(await _result(redis, task)).status for task in tasks] == [TaskStatus.DONE] * 2
    assert await _pending(redis) == 0


async def test_takes_one_task_per_iteration(
    worker: InferenceWorker, submit: Submit, redis: fakeredis.FakeAsyncRedis
) -> None:
    first, second = await submit(make_image(1)), await submit(make_image(2))

    assert await worker.run_once()

    assert await _result(redis, first) is not None
    assert await _result(redis, second) is None
    assert await _pending(redis) == 0  # the second one is not even read yet


async def test_drains_abandoned_tasks_before_waiting_for_the_interval(
    embedder: StubEmbedder,
    qdrant: AsyncQdrantClient,
    submit: Submit,
    redis: fakeredis.FakeAsyncRedis,
    tmp_path: Path,
) -> None:
    worker = _worker(embedder, _settings(tmp_path, claim_interval_s=3600))
    await worker.start()
    tasks = [await submit(make_image(seed)) for seed in range(2)]
    for _ in tasks:
        await read_one(TASK_STREAM, INFERENCE_GROUP, "dead", EmbeddingTask, block_ms=10)

    assert await worker.run_once()
    assert await worker.run_once()

    assert [(await _result(redis, task)).status for task in tasks] == [TaskStatus.DONE] * 2
    assert await _pending(redis) == 0


async def test_run_touches_the_heartbeat_and_stops_on_request(
    worker: InferenceWorker, submit: Submit, redis: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = await submit(make_image(1))
    original = worker.run_once

    async def once_then_stop() -> bool:
        worker.stop()
        return await original()

    monkeypatch.setattr(worker, "run_once", once_then_stop)
    await worker.run()

    assert (await _result(redis, task)).status == TaskStatus.DONE
    assert worker._settings.heartbeat_path.exists()

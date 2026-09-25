import fakeredis
import pytest
from sqlalchemy.exc import OperationalError

from backend.src.services import search as search_service
from backend.tests.conftest import FakeSession
from backend.tests.helpers import BOX_JSON, make_image, make_result
from common.src.configs.constants import RESULT_KEY, TASK_KEY, TASK_STREAM
from common.src.configs.schemas import TaskStatus
from common.src.exceptions import ValidationError


async def test_submit_stores_queues_and_records(
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
    objects: dict[str, bytes],
) -> None:
    task_id = await search_service.submit(session, data=make_image(), filename="q.png", bbox_raw=BOX_JSON, top_k=5)

    assert objects[f"queries/{task_id}.png"]
    assert await redis.get(TASK_KEY.format(task_id=task_id)) == TaskStatus.PENDING.value
    assert len(await redis.xrange(TASK_STREAM)) == 1
    assert len(session.executed) == 1


async def test_submit_survives_a_dead_database(
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
    objects: dict[str, bytes],  # noqa: ARG001  # patches the storage module
) -> None:
    """Postgres holds metadata about the search, not the search: an outage costs a row."""
    session.fail_with = OperationalError("insert", {}, Exception("connection refused"))

    task_id = await search_service.submit(session, data=make_image(), filename="q.png", bbox_raw=BOX_JSON, top_k=5)

    assert await redis.get(TASK_KEY.format(task_id=task_id)) == TaskStatus.PENDING.value
    assert session.rollbacks == 1


async def test_submit_validates_before_writing_anything(
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
    objects: dict[str, bytes],
) -> None:
    with pytest.raises(ValidationError):
        await search_service.submit(session, data=b"not an image", filename="q.png", bbox_raw=BOX_JSON, top_k=5)

    assert objects == {}
    assert await redis.xlen(TASK_STREAM) == 0
    assert session.executed == []


async def test_fetch_result_without_a_key_touches_nothing(
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,  # noqa: ARG001  # the service reads through it
) -> None:
    assert await search_service.fetch_result(session, "missing") is None
    assert session.executed == []


async def test_fetch_result_parses_and_records(
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
) -> None:
    published = make_result(count=2)
    await redis.set(RESULT_KEY.format(task_id=published.task_id), published.model_dump_json())

    result = await search_service.fetch_result(session, published.task_id)

    assert result is not None
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    # The update runs; _NoRows makes it match nothing, so the candidate insert is skipped.
    assert len(session.executed) == 1


async def test_fetch_result_survives_a_dead_database(
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
) -> None:
    published = make_result()
    await redis.set(RESULT_KEY.format(task_id=published.task_id), published.model_dump_json())
    session.fail_with = OperationalError("update", {}, Exception("connection refused"))

    result = await search_service.fetch_result(session, published.task_id)

    assert result is not None
    assert result.task_id == published.task_id
    assert session.rollbacks == 1

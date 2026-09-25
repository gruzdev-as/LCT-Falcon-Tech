from http import HTTPStatus

import fakeredis
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError

from backend.tests.conftest import FakeSession
from backend.tests.helpers import BOX_JSON, make_image, make_result
from common.src.configs.constants import RESULT_KEY, TASK_KEY, TASK_TTL_SECONDS
from common.src.configs.schemas import TaskStatus

ENDPOINT = "/api/v1/search"


def _form(data: bytes | None = None, bbox: str = BOX_JSON, top_k: int = 5) -> dict:
    return {
        "files": {"file": ("q.png", data if data is not None else make_image(), "image/png")},
        "data": {"bbox": bbox, "top_k": str(top_k)},
    }


async def test_submit_returns_a_task_id_and_commits(client: AsyncClient, session: FakeSession) -> None:
    response = await client.post(ENDPOINT, **_form())

    assert response.status_code == HTTPStatus.ACCEPTED
    body = response.json()
    assert len(body["task_id"]) == 32
    assert body["status"] == TaskStatus.PENDING.value
    assert session.commits == 1


async def test_submit_rejects_a_broken_bbox(client: AsyncClient, session: FakeSession) -> None:
    response = await client.post(ENDPOINT, **_form(bbox="not json"))

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["error"] == "validation_error"
    assert session.commits == 0


async def test_unknown_task_is_404(client: AsyncClient, session: FakeSession) -> None:
    response = await client.get(f"{ENDPOINT}/nope")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert session.commits == 1


async def test_live_task_is_202(
    client: AsyncClient,
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
) -> None:
    await redis.set(TASK_KEY.format(task_id="live"), TaskStatus.PENDING.value, ex=TASK_TTL_SECONDS)

    response = await client.get(f"{ENDPOINT}/live")

    assert response.status_code == HTTPStatus.ACCEPTED
    assert response.json()["status"] == TaskStatus.PROCESSING.value
    assert session.commits == 1


async def test_finished_task_returns_candidates(
    client: AsyncClient,
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
) -> None:
    published = make_result(count=3)
    await redis.set(RESULT_KEY.format(task_id=published.task_id), published.model_dump_json())

    response = await client.get(f"{ENDPOINT}/{published.task_id}")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert [candidate["rank"] for candidate in body["candidates"]] == [1, 2, 3]
    assert body["rejected"] is False
    assert session.commits == 1


async def test_a_dead_database_never_becomes_a_500(
    client: AsyncClient,
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,
) -> None:
    """The outage policy, end to end: bookkeeping degrades, the API does not."""
    published = make_result()
    await redis.set(RESULT_KEY.format(task_id=published.task_id), published.model_dump_json())
    session.fail_with = OperationalError("statement", {}, Exception("connection refused"))

    submitted = await client.post(ENDPOINT, **_form())
    fetched = await client.get(f"{ENDPOINT}/{published.task_id}")

    assert submitted.status_code == HTTPStatus.ACCEPTED
    assert fetched.status_code == HTTPStatus.OK

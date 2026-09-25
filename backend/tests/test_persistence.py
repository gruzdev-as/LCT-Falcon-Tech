"""Search persistence against a real Postgres.

`on_conflict_do_nothing` and the row-lock re-check behind the idempotent finalize are
dialect behaviour, so there is nothing useful to learn from running this on anything
else. Point ``POSTGRES_TEST_DSN`` at a throwaway database to enable it:

    POSTGRES_TEST_DSN=postgresql+asyncpg://falcon:falcon@localhost:5432/falcon_test
"""

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from backend.src.db.searches import finalize_search, insert_query
from backend.tests.helpers import BOX, make_result
from common.src.configs.schemas import TaskStatus
from common.src.db.base import Base
from common.src.db.models import SearchCandidate, SearchQuery

DSN = os.getenv("POSTGRES_TEST_DSN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="POSTGRES_TEST_DSN is not set"),
]

type Record = Callable[..., Awaitable[str]]


@pytest.fixture
async def engine() -> AsyncGenerator[AsyncEngine]:
    """A clean schema per test. create_all is sanctioned for tests only."""
    active = create_async_engine(DSN)
    async with active.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield active
    await active.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    async with async_sessionmaker(engine, expire_on_commit=False)() as active:
        yield active


@pytest.fixture
def record(session: AsyncSession) -> Record:
    """Write one query row the way ``submit`` does, and commit it."""

    async def write(task_id: str | None = None) -> str:
        task_id = task_id or uuid.uuid4().hex
        await insert_query(
            session,
            task_id=task_id,
            image_path=f"queries/{task_id}.png",
            bbox=BOX,
            top_k=5,
            image_width=128,
            image_height=96,
            image_format="PNG",
            content_type="image/png",
            size_bytes=4096,
        )
        await session.commit()
        return task_id

    return write


async def _count_candidates(session: AsyncSession, task_id: str) -> int:
    statement = select(func.count()).select_from(SearchCandidate).where(SearchCandidate.task_id == task_id)
    return int(await session.scalar(statement) or 0)


async def test_insert_query_writes_every_column(session: AsyncSession, record: Record) -> None:
    task_id = await record()

    row = await session.scalar(select(SearchQuery).where(SearchQuery.task_id == task_id))

    assert row is not None
    assert (row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height) == (BOX.x, BOX.y, BOX.width, BOX.height)
    assert (row.image_width, row.image_height, row.image_format) == (128, 96, "PNG")
    assert row.content_type == "image/png"
    assert row.size_bytes == 4096
    # The value, not the member name: this is what values_callable protects.
    assert row.status.value == "pending"
    assert row.finalized_at is None


async def test_insert_query_twice_is_a_noop(session: AsyncSession, record: Record) -> None:
    task_id = await record()
    await record(task_id)

    statement = select(func.count()).select_from(SearchQuery).where(SearchQuery.task_id == task_id)
    assert await session.scalar(statement) == 1


async def test_finalize_writes_outcome_and_candidates(session: AsyncSession, record: Record) -> None:
    task_id = await record()
    result = make_result(task_id, count=4)

    assert await finalize_search(session, result) is True
    await session.commit()

    row = await session.get(SearchQuery, task_id)
    assert row is not None
    assert row.status is TaskStatus.DONE
    assert row.candidate_count == 4
    assert row.top_score == result.top_score
    assert row.model_name == "stub"
    assert row.latency_ms == result.latency_ms
    assert row.finished_at is not None
    assert row.finalized_at is not None

    ranks = await session.scalars(
        select(SearchCandidate.rank).where(SearchCandidate.task_id == task_id).order_by(SearchCandidate.rank)
    )
    assert list(ranks) == [1, 2, 3, 4]


async def test_finalize_is_idempotent(session: AsyncSession, record: Record) -> None:
    task_id = await record()
    result = make_result(task_id, count=3)

    assert await finalize_search(session, result) is True
    await session.commit()

    assert await finalize_search(session, result) is False
    await session.commit()

    assert await _count_candidates(session, task_id) == 3


async def test_concurrent_finalize_writes_once(
    session: AsyncSession,
    engine: AsyncEngine,
    record: Record,
) -> None:
    """Two replicas answering the same poll: the row lock decides, not a SELECT."""
    task_id = await record()
    result = make_result(task_id, count=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def attempt() -> bool:
        async with factory() as own:
            won = await finalize_search(own, result)
            await own.commit()
            return won

    outcomes = await asyncio.gather(attempt(), attempt())

    assert sorted(outcomes) == [False, True]
    assert await _count_candidates(session, task_id) == 3


async def test_finalize_without_a_query_row_returns_false(session: AsyncSession) -> None:
    """Postgres was down at POST and up at GET: the outcome is dropped, not orphaned."""
    result = make_result(count=2)

    assert await finalize_search(session, result) is False
    await session.commit()

    assert await _count_candidates(session, result.task_id) == 0


async def test_rejected_result_stores_no_candidates(session: AsyncSession, record: Record) -> None:
    task_id = await record()

    await finalize_search(session, make_result(task_id, rejected=True))
    await session.commit()

    row = await session.get(SearchQuery, task_id)
    assert row is not None
    assert row.rejected is True
    assert row.candidate_count == 0
    assert await _count_candidates(session, task_id) == 0


async def test_failed_result_stores_the_error(session: AsyncSession, record: Record) -> None:
    task_id = await record()

    await finalize_search(session, make_result(task_id, status=TaskStatus.FAILED, error="bbox degenerate"))
    await session.commit()

    row = await session.get(SearchQuery, task_id)
    assert row is not None
    assert row.status is TaskStatus.FAILED
    assert row.error == "bbox degenerate"


async def test_presigned_url_is_not_persisted(session: AsyncSession, record: Record) -> None:
    """image_url expires within the hour; image_path is what regenerates it."""
    task_id = await record()
    result = make_result(task_id, count=1)
    assert result.candidates[0].image_url is not None

    await finalize_search(session, result)
    await session.commit()

    row = await session.scalar(select(SearchCandidate).where(SearchCandidate.task_id == task_id))
    assert row is not None
    assert row.image_path == "gallery/img-1.png"
    assert "X-Amz-Signature" not in str(row.__dict__.values())


async def test_deleting_a_query_removes_its_candidates(session: AsyncSession, record: Record) -> None:
    """The cascade is what makes a future retention job one statement."""
    task_id = await record()
    await finalize_search(session, make_result(task_id, count=3))
    await session.commit()

    await session.execute(delete(SearchQuery).where(SearchQuery.task_id == task_id))
    await session.commit()

    assert await _count_candidates(session, task_id) == 0

"""Metadata step against a real Postgres.

`on_conflict_do_update` is dialect-specific, so there is nothing useful to learn
from running this on anything else. Point ``POSTGRES_TEST_DSN`` at a throwaway
database to enable it:

    POSTGRES_TEST_DSN=postgresql+asyncpg://falcon:falcon@localhost:5432/falcon_test
"""

import os
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from common.src.db.base import Base
from common.src.db.models import GalleryImage
from init.src.gallery import Gallery
from init.src.steps.images import object_key
from init.src.steps.metadata import count_rows, sync_gallery

DSN = os.getenv("POSTGRES_TEST_DSN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="POSTGRES_TEST_DSN is not set"),
]


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession]:
    """A clean schema per test. create_all is sanctioned for tests only."""
    engine = create_async_engine(DSN)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as active:
        yield active
    await engine.dispose()


async def test_writes_every_manifest_row(session: AsyncSession, gallery: Gallery) -> None:
    written = await sync_gallery(session, gallery)
    await session.commit()

    assert written == gallery.count
    assert await count_rows(session, gallery.version) == gallery.count


async def test_a_second_run_writes_nothing(session: AsyncSession, gallery: Gallery) -> None:
    await sync_gallery(session, gallery)
    await session.commit()

    assert await sync_gallery(session, gallery) == 0


async def test_rerunning_refreshes_instead_of_failing(session: AsyncSession, gallery: Gallery) -> None:
    """Same primary keys on every run: an insert-only step would break on the second."""
    await sync_gallery(session, gallery)
    await session.commit()

    await sync_gallery(session, gallery, force=True)
    await session.commit()

    assert await count_rows(session, gallery.version) == gallery.count


async def test_stores_the_object_key_not_the_manifest_path(session: AsyncSession, gallery: Gallery) -> None:
    await sync_gallery(session, gallery)
    await session.commit()

    entry = gallery.entries[0]
    row = await session.scalar(select(GalleryImage).where(GalleryImage.image_id == entry.image_id))

    assert row is not None
    assert row.image_path == object_key(entry)
    assert row.vehicle_id == entry.vehicle_id

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from common.src.db.base import Base
from common.src.db.config import PostgresConfig

# Importing the models registers them on Base.metadata; without it create_schema will create nothing at all.
import common.src.db.models  # noqa: F401  # isort: skip

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the shared engine, creating it on first use."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        config = PostgresConfig()
        _engine = create_async_engine(config.dsn, pool_size=config.pool_size, echo=config.echo, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the shared session factory."""
    global _sessionmaker  # noqa: PLW0603
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def create_schema() -> None:
    """Create any table that does not exist yet, leaving existing ones untouched."""
    async with get_engine().begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Database schema is present")


async def close_engine() -> None:
    """Dispose of the engine. Called on application shutdown."""
    global _engine, _sessionmaker  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession]:
    """One session for one unit of work, committed on a clean exit."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

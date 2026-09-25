from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from common.src.db.session import get_sessionmaker


async def db_session() -> AsyncGenerator[AsyncSession]:
    """Yield one session per request; the router commits it on the happy path."""
    async with get_sessionmaker()() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(db_session)]

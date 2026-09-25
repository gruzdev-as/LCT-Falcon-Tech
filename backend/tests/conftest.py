from collections.abc import AsyncIterator

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.dependencies import db_session
from backend.app.server import app
from common.src.exceptions import NotFoundError
from common.src.redis import client as redis_module
from common.src.storage import client as storage_module


class FakeSession:
    """Records statements instead of running them, and can be told to fail.

    Enough to unit-test the service and the routers without a database, including the
    degraded path: set ``fail_with`` and every ``execute`` raises it.
    """

    def __init__(self) -> None:
        self.executed: list[object] = []
        self.commits = 0
        self.rollbacks = 0
        self.fail_with: Exception | None = None

    async def execute(self, statement: object) -> object:
        """Record the statement, or raise whatever ``fail_with`` holds."""
        if self.fail_with is not None:
            raise self.fail_with
        self.executed.append(statement)
        return _NoRows()

    async def commit(self) -> None:
        """Count the commit the router makes at its boundary."""
        self.commits += 1

    async def rollback(self) -> None:
        """Count the rollback the service makes when a statement failed."""
        self.rollbacks += 1


class _NoRows:
    """Stands in for a CursorResult that matched nothing."""

    rowcount = 0


@pytest.fixture
async def redis(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    monkeypatch.setattr(redis_module, "_redis", client)
    yield client
    await client.aclose()


@pytest.fixture
def objects(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """In-memory object store standing in for S3."""
    store: dict[str, bytes] = {}

    async def save(key: str, data: bytes, content_type: str = "image/jpeg") -> str:  # noqa: ARG001
        store[key] = data
        return key

    async def load(key: str) -> bytes:
        if key not in store:
            msg = f"image {key} does not exist"
            raise NotFoundError(msg)
        return store[key]

    async def ensure_bucket() -> None:
        return None

    monkeypatch.setattr(storage_module, "save", save)
    monkeypatch.setattr(storage_module, "load", load)
    monkeypatch.setattr(storage_module, "ensure_bucket", ensure_bucket)
    monkeypatch.setattr(storage_module, "url_for", lambda key, ttl=None: f"http://s3.test/{key}")
    return store


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
async def client(
    session: FakeSession,
    redis: fakeredis.FakeAsyncRedis,  # noqa: ARG001  # patches the module-level client
    objects: dict[str, bytes],  # noqa: ARG001  # patches the storage module
) -> AsyncIterator[AsyncClient]:
    """The app over ASGITransport, which does not run the lifespan — so no real engine."""
    app.dependency_overrides[db_session] = lambda: session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()

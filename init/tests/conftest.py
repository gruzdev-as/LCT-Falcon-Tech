from collections.abc import AsyncGenerator, AsyncIterator, Generator
from pathlib import Path

import fakeredis
import pytest
from qdrant_client import AsyncQdrantClient

from common.src.qdrant import client as qdrant_module
from common.src.redis import client as redis_module
from common.src.storage import client as storage_module
from init.src.gallery import Gallery, load_gallery
from init.tests.helpers import write_gallery


@pytest.fixture
def gallery(tmp_path: Path) -> Gallery:
    """A small, valid gallery on disk."""
    return load_gallery(write_gallery(tmp_path / "gallery", count=6))


@pytest.fixture
async def qdrant_client(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[AsyncQdrantClient]:
    """In-memory Qdrant, shared with the module-level client the steps use.

    Left empty on purpose: the workers create the collection, sized by their model.
    """
    client = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(qdrant_module, "_qdrant", client)
    yield client
    await client.close()


@pytest.fixture
async def redis(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    monkeypatch.setattr(redis_module, "_redis", client)
    yield client
    await client.aclose()


@pytest.fixture
def objects(monkeypatch: pytest.MonkeyPatch) -> Generator[dict[str, bytes]]:
    """Dict-backed object storage, matching the fake used by the inference suite."""
    store: dict[str, bytes] = {}

    async def save(key: str, data: bytes, content_type: str = "image/jpeg") -> str:  # noqa: ARG001
        store[key] = data
        return key

    async def list_keys(prefix: str) -> set[str]:
        return {key for key in store if key.startswith(prefix)}

    async def ensure_bucket() -> None:
        return None

    monkeypatch.setattr(storage_module, "save", save)
    monkeypatch.setattr(storage_module, "list_keys", list_keys)
    monkeypatch.setattr(storage_module, "ensure_bucket", ensure_bucket)
    yield store

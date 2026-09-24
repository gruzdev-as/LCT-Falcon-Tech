from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from common.src.qdrant import client as qdrant_module
from common.src.storage import client as storage_module
from init.src.bundle import Bundle, load_bundle
from init.tests.helpers import write_bundle


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    """A small, valid bundle on disk."""
    return load_bundle(write_bundle(tmp_path / "artifacts", count=6))


@pytest.fixture
async def qdrant_client(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[AsyncQdrantClient]:
    """In-memory Qdrant, shared with the module-level client the steps use.

    Left empty on purpose: creating the collection is the bootstrap's job.
    """
    client = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(qdrant_module, "_qdrant", client)
    yield client
    await client.close()


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

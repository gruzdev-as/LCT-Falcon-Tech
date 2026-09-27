import uuid
from collections.abc import AsyncIterator

import fakeredis
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from common.src.configs.constants import GALLERY_COLLECTION, TASK_KEY, TASK_STREAM, TASK_TTL_SECONDS
from common.src.configs.schemas import BBox, EmbeddingTask, TaskStatus
from common.src.exceptions import NotFoundError
from common.src.qdrant import client as qdrant_module
from common.src.redis import client as redis_module
from common.src.storage import client as storage_module
from inference.src.pipeline.postprocess import l2_normalize
from inference.tests.helpers import BOX, DIM, FakeEmbedder, GalleryAdd, Submit, make_task


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=DIM)


@pytest.fixture
async def redis(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    monkeypatch.setattr(redis_module, "_redis", client)
    yield client
    await client.aclose()


@pytest.fixture
async def qdrant(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(qdrant_module, "_qdrant", client)
    await qdrant_module.ensure_collection(GALLERY_COLLECTION, DIM)
    yield client
    await client.close()


@pytest.fixture
def objects(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """In-memory object store standing in for S3."""
    store: dict[str, bytes] = {}

    async def load(key: str) -> bytes:
        if key not in store:
            msg = f"image {key} does not exist"
            raise NotFoundError(msg)
        return store[key]

    monkeypatch.setattr(storage_module, "load", load)
    monkeypatch.setattr(storage_module, "url_for", lambda key, ttl=None: f"http://s3.test/{key}")
    return store


@pytest.fixture
def gallery_add(qdrant: AsyncQdrantClient, embedder: FakeEmbedder) -> GalleryAdd:
    """Index an image the way gallery ingestion will: same crop, same model."""

    async def add(data: bytes, bbox: BBox = BOX, **payload: str) -> str:
        vector = l2_normalize(embedder.embed(data, bbox))
        point_id = str(uuid.uuid4())
        payload.setdefault("image_path", f"gallery/{point_id}.png")
        await qdrant.upsert(GALLERY_COLLECTION, [PointStruct(id=point_id, vector=vector.tolist(), payload=payload)])
        return point_id

    return add


@pytest.fixture
def submit(redis: fakeredis.FakeAsyncRedis, objects: dict[str, bytes]) -> Submit:
    """Queue a task exactly as the backend's ``submit`` does."""

    async def queue(data: bytes | None = None, *, live: bool = True, top_k: int = 5) -> EmbeddingTask:
        task = make_task(image_path=f"queries/{uuid.uuid4().hex}.png", top_k=top_k)
        if data is not None:
            objects[task.image_path] = data
        if live:
            await redis.set(TASK_KEY.format(task_id=task.task_id), TaskStatus.PENDING.value, ex=TASK_TTL_SECONDS)
        await redis_module.publish(TASK_STREAM, task)
        return task

    return queue

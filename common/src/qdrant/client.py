import logging
from collections.abc import Sequence
from dataclasses import asdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from common.src.exceptions import StorageError
from common.src.qdrant.config import QdrantConfig

logger = logging.getLogger(__name__)

_qdrant: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    """Return the shared Qdrant client, creating it on first use."""
    global _qdrant  # noqa: PLW0603
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(**asdict(QdrantConfig()))
    return _qdrant


async def close_qdrant() -> None:
    """Close the shared client. Called on shutdown."""
    global _qdrant  # noqa: PLW0603
    if _qdrant is not None:
        await _qdrant.close()
        _qdrant = None


async def ensure_collection(name: str, dim: int) -> None:
    """Create a collection if it does not exist yet, idempotently.

    Raises:
        StorageError: if the collection exists with a different vector size
    """
    client = get_qdrant()
    if not await client.collection_exists(name):
        try:
            await client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            logger.info("Created Qdrant collection %s (dim=%d)", name, dim)
        except Exception:
            if not await client.collection_exists(name):
                raise
            logger.debug("Collection %s was created concurrently", name)

    vectors = (await client.get_collection(name)).config.params.vectors
    size = vectors.size if isinstance(vectors, VectorParams) else None
    if size != dim:
        msg = f"collection {name} holds {size}-dim vectors, the model produces {dim}"
        raise StorageError(msg, details={"collection": name, "expected": dim, "actual": size})


async def recreate_collection(name: str, dim: int) -> None:
    """Drop the collection and build it empty again."""
    client = get_qdrant()
    await client.delete_collection(name)
    await client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    logger.info("Recreated Qdrant collection %s (dim=%d)", name, dim)


async def count_points(collection: str, *, where: tuple[str, str] | None = None) -> int:
    """Return how many points the collection holds, or 0 if it does not exist.

    Args:
        collection: collection name.
        where: optional ``(payload field, value)`` the counted points must match.
    """
    client = get_qdrant()
    if not await client.collection_exists(collection):
        return 0
    condition = None
    if where is not None:
        key, value = where
        condition = Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])
    return (await client.count(collection, count_filter=condition, exact=True)).count


async def payload_values(collection: str, field: str, *, page: int = 1024) -> dict[str, object]:
    """Map every point id to one payload field, or return {} if the collection does not exist."""
    client = get_qdrant()
    if not await client.collection_exists(collection):
        return {}
    values: dict[str, object] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            collection, limit=page, offset=offset, with_payload=[field], with_vectors=False
        )
        values.update({str(point.id): (point.payload or {}).get(field) for point in points})
        if offset is None:
            return values


async def delete_points(collection: str, ids: Sequence[str]) -> None:
    """Remove points by id, waiting until they are gone from search."""
    if ids:
        await get_qdrant().delete(collection, points_selector=PointIdsList(points=list(ids)), wait=True)


async def delete_collection(name: str) -> None:
    """Drop a collection if it exists."""
    client = get_qdrant()
    if await client.collection_exists(name):
        await client.delete_collection(name)
        logger.info("Dropped Qdrant collection %s", name)


async def upsert_points(collection: str, points: Sequence[PointStruct]) -> None:
    """Write one batch of points, waiting until they are searchable."""
    if not points:
        return
    await get_qdrant().upsert(collection, list(points), wait=True)


async def search(
    collection: str, vector: Sequence[float], limit: int, *, with_vectors: bool = False
) -> list[ScoredPoint]:
    """Find the nearest neighbours of one vector, optionally with their stored vectors."""
    response = await get_qdrant().query_points(
        collection, query=list(vector), limit=limit, with_payload=True, with_vectors=with_vectors
    )
    return response.points

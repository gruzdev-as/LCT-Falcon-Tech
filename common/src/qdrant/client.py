import logging
from collections.abc import Sequence
from dataclasses import asdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

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


async def count_points(collection: str) -> int:
    """Return how many points the collection holds, or 0 if it does not exist."""
    client = get_qdrant()
    if not await client.collection_exists(collection):
        return 0
    return (await client.count(collection, exact=True)).count


async def upsert_points(collection: str, points: Sequence[PointStruct]) -> None:
    """Write one batch of points, waiting until they are searchable."""
    if not points:
        return
    await get_qdrant().upsert(collection, list(points), wait=True)


async def search(collection: str, vector: Sequence[float], limit: int) -> list[ScoredPoint]:
    """Find the nearest neighbours of one vector."""
    response = await get_qdrant().query_points(collection, query=list(vector), limit=limit, with_payload=True)
    return response.points

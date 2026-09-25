import logging
from uuid import uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.db import searches
from backend.src.services.validation import ImageInfo, parse_bbox, validate_bbox, validate_image
from common.src.configs.constants import (
    QUERY_PREFIX,
    RESULT_KEY,
    TASK_KEY,
    TASK_STREAM,
    TASK_TTL_SECONDS,
)
from common.src.configs.schemas import BBox, EmbeddingTask, SearchResult, TaskStatus
from common.src.redis.client import get_redis, publish
from common.src.storage import client as storage

logger = logging.getLogger(__name__)


async def submit(
    session: AsyncSession,
    *,
    data: bytes,
    filename: str,
    bbox_raw: str,
    top_k: int,
    declared_content_type: str | None = None,
) -> str:
    """Accept an image and queue it for re-identification.

    Args:
        session: open session; not committed here.
        data: raw bytes of the uploaded image.
        filename: original filename, used for the object key's extension.
        bbox_raw: bbox as a JSON string from the multipart form.
        top_k: how many candidates inference should return.
        declared_content_type: Content-Type of the upload, if the client sent one.

    Returns:
        The task id the caller polls for a result.

    Raises:
        ValidationError: the image or the bbox failed validation.
        StorageError: the original could not be stored.
    """
    info = validate_image(data, declared_content_type)
    bbox = validate_bbox(parse_bbox(bbox_raw), info)

    task_id = uuid4().hex
    image_path = storage.make_path(filename, prefix=QUERY_PREFIX, name=task_id)

    await storage.save(image_path, data, content_type=info.content_type)
    await get_redis().set(TASK_KEY.format(task_id=task_id), TaskStatus.PENDING.value, ex=TASK_TTL_SECONDS)

    task = EmbeddingTask(task_id=task_id, image_path=image_path, bbox=bbox, top_k=top_k)
    await publish(TASK_STREAM, task)
    await _record_query(session, task_id=task_id, image_path=image_path, bbox=bbox, top_k=top_k, info=info)

    logger.info("Queued task %s for image %s (top_k=%d)", task_id, image_path, top_k)
    return task_id


async def fetch_result(session: AsyncSession, task_id: str) -> SearchResult | None:
    """Return a finished result, recording it in Postgres the first time it is seen.

    Args:
        session: open session; not committed here.
        task_id: the id the client is polling.

    Returns:
        The finished result, or None if it is not ready.
    """
    raw = await get_redis().get(RESULT_KEY.format(task_id=task_id))
    if raw is None:
        return None

    result = SearchResult.model_validate_json(raw)
    await _record_result(session, result)
    return result


async def task_exists(task_id: str) -> bool:
    """Report whether the task is known and still in process."""
    return bool(await get_redis().exists(TASK_KEY.format(task_id=task_id)))


async def _record_query(
    session: AsyncSession,
    *,
    task_id: str,
    image_path: str,
    bbox: BBox,
    top_k: int,
    info: ImageInfo,
) -> None:
    """Persist the request row, best effort: Postgres holds metadata, not the search."""
    try:
        await searches.insert_query(
            session,
            task_id=task_id,
            image_path=image_path,
            bbox=bbox,
            top_k=top_k,
            image_width=info.width,
            image_height=info.height,
            image_format=info.image_format,
            content_type=info.content_type,
            size_bytes=info.size_bytes,
        )
    except SQLAlchemyError:
        await session.rollback()
        logger.warning("Could not record search query %s", task_id, exc_info=True)


async def _record_result(session: AsyncSession, result: SearchResult) -> None:
    """Persist the outcome and its candidates, best effort and only once."""
    try:
        if await searches.finalize_search(session, result):
            logger.info("Recorded result for task %s (%d candidates)", result.task_id, len(result.candidates))
    except SQLAlchemyError:
        await session.rollback()
        logger.warning("Could not record search result %s", result.task_id, exc_info=True)

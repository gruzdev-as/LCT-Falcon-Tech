import logging
from uuid import uuid4

from backend.src.services.validation import parse_bbox, validate_bbox, validate_image
from common.src.configs.constants import (
    QUERY_PREFIX,
    RESULT_KEY,
    TASK_KEY,
    TASK_STREAM,
    TASK_TTL_SECONDS,
)
from common.src.configs.schemas import EmbeddingTask, SearchResult, TaskStatus
from common.src.redis.client import get_redis, publish
from common.src.storage import client as storage

logger = logging.getLogger(__name__)


async def submit(
    *,
    data: bytes,
    filename: str,
    bbox_raw: str,
    top_k: int,
    declared_content_type: str | None = None,
) -> str:
    """Accept an image and queue it for re-identification.

    Args:
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

    logger.info("Queued task %s for image %s (top_k=%d)", task_id, image_path, top_k)
    return task_id


async def fetch_result(task_id: str) -> SearchResult | None:
    """Return a finished result, or None if it is not ready."""
    raw = await get_redis().get(RESULT_KEY.format(task_id=task_id))
    return None if raw is None else SearchResult.model_validate_json(raw)


async def task_exists(task_id: str) -> bool:
    """Report whether the task is known and still in process."""
    return bool(await get_redis().exists(TASK_KEY.format(task_id=task_id)))

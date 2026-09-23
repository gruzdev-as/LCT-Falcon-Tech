from common.src.configs.constants import RESULT_KEY, RESULT_TTL_SECONDS, TASK_KEY
from common.src.configs.schemas import SearchResult, TaskStatus
from common.src.redis.client import get_redis

# Inference's side of the task-state protocol. The backend's side (setting the
# pending marker, reading the result) lives in backend/src/services/search.py.


async def mark_processing(task_id: str) -> bool:
    """Flip a live task's marker to ``processing``, keeping its TTL.

    Returns:
        False if the marker is gone: the task expired, or its result is already
        written. Either way there is nobody left to compute a result for.
    """
    key = TASK_KEY.format(task_id=task_id)
    return bool(await get_redis().set(key, TaskStatus.PROCESSING.value, xx=True, keepttl=True))


async def write_result(result: SearchResult) -> None:
    """Publish a finished result and retire the task marker in one transaction."""
    async with get_redis().pipeline(transaction=True) as pipe:
        pipe.set(RESULT_KEY.format(task_id=result.task_id), result.model_dump_json(), ex=RESULT_TTL_SECONDS)
        pipe.delete(TASK_KEY.format(task_id=result.task_id))
        await pipe.execute()

import logging
from dataclasses import asdict
from typing import cast

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from common.src.configs.constants import STREAM_MAXLEN, STREAM_PAYLOAD_FIELD
from common.src.redis.config import RedisConfig, StreamConfig, StreamEntry, StreamReply

logger = logging.getLogger(__name__)

_redis: Redis | None = None


def get_redis() -> Redis:
    """Return the shared Redis client, creating it on first use."""
    global _redis  # noqa: PLW0603
    if _redis is None:
        _redis = Redis(**asdict(RedisConfig()))
    return _redis


async def close_redis() -> None:
    """Close the shared client. Called on application shutdown."""
    global _redis  # noqa: PLW0603
    if _redis is not None:
        await _redis.aclose()
        _redis = None


async def init_redis_streams() -> None:
    """Create the consumer group on every stream if it doesn't exist, idempotently.

    The group starts at id 0, so entries published before it existed — gallery tasks
    init queued before the workers came up — are still delivered.
    """
    config = StreamConfig()
    for stream in config.stream_names:
        try:
            await get_redis().xgroup_create(stream, config.group_name, id="0", mkstream=True)
            logger.info("Initialized consumer group %s for stream: %s", config.group_name, stream)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            logger.debug("Consumer group %s already exists on stream: %s", config.group_name, stream)


async def stream_backlog(stream: str, group: str) -> int:
    """Entries of a stream the group has not finished: never delivered plus delivered but unacked.

    Returns 0 when the stream or the group does not exist yet.
    """
    redis = get_redis()
    if not await redis.exists(stream):
        return 0
    for info in await redis.xinfo_groups(stream):
        if info["name"] == group:
            return int(info.get("lag") or 0) + int(info["pending"])
    return int(await redis.xlen(stream))


async def delete_stream(stream: str) -> None:
    """Drop a stream with its groups and pending entries."""
    await get_redis().delete(stream)


async def publish(stream: str, message: BaseModel) -> str:
    """Append a message to a stream."""
    entry_id = await get_redis().xadd(
        stream,
        {STREAM_PAYLOAD_FIELD: message.model_dump_json()},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )
    return cast("str", entry_id)  # Just for typing based on our contracts


async def read_one[T: BaseModel](
    stream: str,
    group: str,
    consumer: str,
    model: type[T],
    *,
    block_ms: int | None = 5_000,
) -> tuple[str, T] | None:
    """Read the next new entry as part of a consumer group.

    Args:
        stream: stream name.
        group: consumer group name.
        consumer: unique name for this worker within the group.
        model: pydantic model the payload is parsed into.
        block_ms: how long the read waits for an entry; None returns at once.

    Returns:
        (message_id, parsed_payload), or None if nothing arrived in time. An
        unparseable entry is acked and dropped, and None is returned as well.
    """
    reply = await get_redis().xreadgroup(
        groupname=group,
        consumername=consumer,
        streams={stream: ">"},
        count=1,
        block=block_ms,
    )
    entries = [entry for _, batch in cast("StreamReply", reply or []) for entry in batch]
    return await _parse_entry(stream, group, entries[0], model) if entries else None


async def claim_one[T: BaseModel](
    stream: str,
    group: str,
    consumer: str,
    model: type[T],
    *,
    min_idle_ms: int,
) -> tuple[str, T, int] | None:
    """Take over an entry that another consumer read but never acknowledged.

    Args:
        stream: stream name.
        group: consumer group name.
        consumer: the claiming consumer; the entry becomes pending on it.
        model: pydantic model the payload is parsed into.
        min_idle_ms: only an entry pending at least this long is claimed.

    Returns:
        (message_id, parsed_payload, delivery_count), or None if nothing is abandoned.
    """
    redis = get_redis()
    _next_id, claimed, *_deleted = await redis.xautoclaim(
        stream, group, consumer, min_idle_time=min_idle_ms, start_id="0-0", count=1
    )
    if not claimed:
        return None
    parsed = await _parse_entry(stream, group, cast("StreamEntry", claimed[0]), model)
    if parsed is None:
        return None

    # XAUTOCLAIM does not report delivery counts, so look the claimed id up
    message_id, payload = parsed
    pending = await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1)
    deliveries = int(pending[0]["times_delivered"]) if pending else 1
    return message_id, payload, deliveries


async def _parse_entry[T: BaseModel](
    stream: str, group: str, entry: StreamEntry, model: type[T]
) -> tuple[str, T] | None:
    message_id, fields = entry
    try:
        return message_id, model.model_validate_json(fields.get(STREAM_PAYLOAD_FIELD) or "")
    except Exception:
        logger.exception("Dropping malformed entry %s on stream %s", message_id, stream)
        await ack(stream, group, message_id)
        return None


async def ack(stream: str, group: str, message_id: str) -> None:
    """Acknowledge a processed message."""
    await get_redis().xack(stream, group, message_id)

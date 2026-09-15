import logging
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import cast

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from common.src.configs.constants import STREAM_MAXLEN, STREAM_PAYLOAD_FIELD
from common.src.redis.config import RedisConfig, StreamConfig, StreamReply

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
    """Create the consumer group if it doesn't exist idempotently."""
    config = StreamConfig()
    try:
        await get_redis().xgroup_create(config.stream_name, config.group_name, id="0", mkstream=True)
        logger.info("Initialized consumer group %s for stream: %s", config.group_name, config.stream_name)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
        logger.debug("Consumer group %s already exists on stream: %s", config.group_name, config.stream_name)


async def publish(stream: str, message: BaseModel) -> str:
    """Append a message to a stream."""
    entry_id = await get_redis().xadd(
        stream,
        {STREAM_PAYLOAD_FIELD: message.model_dump_json()},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )
    return cast("str", entry_id)  # Just for typing based on our contracts


async def consume[T: BaseModel](
    stream: str,
    group: str,
    consumer: str,
    model: type[T],
    *,
    block_ms: int = 5_000,
    count: int = 10,
) -> AsyncIterator[tuple[str, T]]:
    """Read a stream as part of a consumer group, forever.

    Args:
        stream: stream name.
        group: consumer group name.
        consumer: unique name for this worker within the group.
        model: pydantic model the payload is parsed into.
        block_ms: how long a single read waits for new entries.
        count: how many entries to fetch per read.

    Yields:
        (message_id, parsed_payload) pairs. Unparseable entries are acked and skipped
    """
    redis = get_redis()
    while True:
        reply = await redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
            block=block_ms,
        )
        if not reply:
            continue

        for _stream_name, entries in cast("StreamReply", reply):
            for message_id, fields in entries:
                try:
                    payload = model.model_validate_json(fields.get(STREAM_PAYLOAD_FIELD) or "")
                except Exception:
                    logger.exception("Dropping malformed entry %s on stream %s", message_id, stream)
                    await ack(stream, group, message_id)
                    continue
                yield message_id, payload


async def ack(stream: str, group: str, message_id: str) -> None:
    """Acknowledge a processed message."""
    await get_redis().xack(stream, group, message_id)

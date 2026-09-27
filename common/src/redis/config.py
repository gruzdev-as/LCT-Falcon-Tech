import os
from dataclasses import dataclass, field

from common.src.configs.constants import GALLERY_STREAM, INFERENCE_GROUP, TASK_STREAM

type StreamEntry = tuple[str, dict[str, str]]
type StreamReply = list[tuple[str, list[StreamEntry]]]


@dataclass(frozen=True)
class RedisConfig:
    """Configure Redis connection.

    Field names mirror ``redis.Redis`` keyword arguments so the whole dataclass
    can be splatted into the constructor.
    """

    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: str | None = field(default_factory=lambda: os.getenv("REDIS_PASSWORD") or None)
    # Must stay above the longest blocking stream read (INFERENCE_BLOCK_MS), or an
    # idle XREADGROUP times out on the client before Redis answers with nothing
    socket_timeout: float = field(default_factory=lambda: float(os.getenv("REDIS_SOCKET_TIMEOUT", "10")))
    decode_responses: bool = True


@dataclass(frozen=True)
class StreamConfig:
    """Configure Redis Streams."""

    stream_names: tuple[str, ...] = (TASK_STREAM, GALLERY_STREAM)
    group_name: str = INFERENCE_GROUP

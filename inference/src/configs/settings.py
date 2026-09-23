import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    """Worker configuration, read from ``INFERENCE_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_", extra="ignore")

    embedder: str = Field(default="stub", description="Embedding backend, see models.factory")
    embedding_dim: int = Field(default=512, gt=0)
    reject_threshold: float = Field(default=0.5, ge=0, le=1, description="Score below this yields a rejected result")

    # 0 would mean "block forever" to Redis and make the worker deaf to shutdown.
    # Must stay below REDIS_SOCKET_TIMEOUT, see common.src.redis.config.
    block_ms: int = Field(default=2_000, ge=1, description="How long one stream read waits for tasks")

    claim_idle_ms: int = Field(default=60_000, ge=0, description="Pending this long means its worker is dead")
    claim_interval_s: float = Field(default=15.0, ge=0, description="How often to look for abandoned tasks")
    max_deliveries: int = Field(default=3, ge=1, description="Attempts before a task is failed as poison")

    heartbeat_path: Path = Field(
        default=Path(tempfile.gettempdir()) / "inference.alive",
        description="Touched after every healthy iteration; the container healthcheck reads its mtime",
    )


@lru_cache
def get_settings() -> InferenceSettings:
    """Return the cached worker settings."""
    return InferenceSettings()

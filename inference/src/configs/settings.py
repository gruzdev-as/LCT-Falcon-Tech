from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference.src.configs.constants import HEARTBEAT_PATH, REFUSAL_CONFIG, REFUSAL_MODEL_PATH, WEIGHTS_PATH


class InferenceSettings(BaseSettings):
    """Worker configuration, read from ``INFERENCE_*`` environment variables."""

    # Empty means unset, so compose can pass an optional variable through as ``${VAR:-}``.
    model_config = SettingsConfigDict(env_prefix="INFERENCE_", extra="ignore", env_ignore_empty=True)

    refusal_config: Path = Field(default=REFUSAL_CONFIG, description="Training's refusal preset: kind and thresholds")
    refusal_model_path: Path = Field(default=REFUSAL_MODEL_PATH, description="CatBoost refusal head")
    reject_threshold: float | None = Field(default=None, ge=0, le=1, description="Override: reject below this cosine.")

    weights_path: Path = Field(default=WEIGHTS_PATH, description="Serving checkpoint of the ReID model")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(default="auto", description="auto: cuda if present, else cpu")
    compile: bool = Field(default=False, description="torch.compile the model; honoured on cuda only")
    torch_threads: int = Field(default=0, ge=0, description="CPU threads per replica; 0 leaves torch's default")

    # 0 would mean "block forever" to Redis and make the worker deaf to shutdown.
    # Must stay below REDIS_SOCKET_TIMEOUT, see common.src.redis.config.
    block_ms: int = Field(default=2_000, ge=1, description="How long one stream read waits for tasks")

    claim_idle_ms: int = Field(default=60_000, ge=0, description="Pending this long means its worker is dead")
    claim_interval_s: float = Field(default=15.0, ge=0, description="How often to look for abandoned tasks")
    max_deliveries: int = Field(default=3, ge=1, description="Attempts before a task is failed as poison")

    heartbeat_path: Path = Field(default=HEARTBEAT_PATH, description="Touched after every healthy iteration")


@lru_cache
def get_settings() -> InferenceSettings:
    """Return the cached worker settings."""
    return InferenceSettings()

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InitSettings(BaseSettings):
    """Bootstrap configuration, read from ``INIT_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="INIT_", extra="ignore", protected_namespaces=())

    gallery_dir: Path = Field(default=Path("data/gallery"), description="Mounted gallery: manifest.csv and images/")
    weights_dir: Path = Field(default=Path("data/weights"), description="Where model weights are downloaded")

    model_url: str = Field(default="", description="Direct link to the model weights")
    model_sha256: str = Field(default="", description="Expected sha256; empty skips verification")
    catboost_url: str = Field(default="", description="Direct link to the CatBoost refusal head")
    catboost_sha256: str = Field(default="")
    hf_token: SecretStr = Field(default=SecretStr(""), description="Read token for gated Hugging Face repos")

    force: bool = Field(default=False, description="Redownload and rebuild instead of skipping what is present")
    upload_concurrency: int = Field(default=16, ge=1, le=128, description="Parallel uploads to object storage")

    wait_poll_s: float = Field(default=5.0, gt=0, description="gallery-ready: how often to check the progress")
    wait_stall_s: float = Field(default=300.0, gt=0, description="gallery-ready: warn after this long with no progress")
    wait_timeout_s: float = Field(default=0.0, ge=0, description="gallery-ready: give up after this long; 0 waits on")


@lru_cache
def get_settings() -> InitSettings:
    """Return the cached bootstrap settings."""
    return InitSettings()

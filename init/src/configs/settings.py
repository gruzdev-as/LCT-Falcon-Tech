from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InitSettings(BaseSettings):
    """Bootstrap configuration, read from ``INIT_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="INIT_", extra="ignore", protected_namespaces=())

    artifacts_dir: Path = Field(default=Path("data/gallery"), description="Where the vectors bundle is unpacked")
    weights_dir: Path = Field(default=Path("data/weights"), description="Where model weights are downloaded")

    model_url: str = Field(default="", description="Direct link to the model weights; empty disables the artifact")
    model_sha256: str = Field(default="", description="Expected sha256; empty skips verification")
    catboost_url: str = Field(default="", description="Direct link to the CatBoost head on top of the embeddings")
    catboost_sha256: str = Field(default="")
    images_url: str = Field(default="", description="Direct link to the gallery images archive")
    images_sha256: str = Field(default="")
    vectors_url: str = Field(default="", description="Direct link to the embeddings bundle archive")
    vectors_sha256: str = Field(default="")

    hf_token: SecretStr = Field(default=SecretStr(""), description="Read token for gated Hugging Face repos")

    force: bool = Field(default=False, description="Redownload and rebuild instead of skipping what is present")

    upload_concurrency: int = Field(default=16, ge=1, le=128, description="Parallel uploads to object storage")
    upsert_batch: int = Field(default=256, ge=1, le=4096, description="Points per Qdrant upsert call")

    @property
    def bundle_dir(self) -> Path:
        """Directory holding bundle.json, embeddings.npy and manifest.csv."""
        return self.artifacts_dir


@lru_cache
def get_settings() -> InitSettings:
    """Return the cached bootstrap settings."""
    return InitSettings()

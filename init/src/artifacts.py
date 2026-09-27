import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import anyio.to_thread
import httpx

from common.src.exceptions import ValidationError
from init.src.configs.constants import DOWNLOAD_TIMEOUT_S, HASH_CHUNK_BYTES, HF_HOSTS
from init.src.configs.settings import InitSettings
from init.src.fetch import fetch, filename_from_url

logger = logging.getLogger(__name__)

MODEL = "model"
CATBOOST = "catboost"


@dataclass(frozen=True, slots=True)
class Artifact:
    """One downloadable file of the bootstrap."""

    name: str
    url: str
    sha256: str
    target_dir: Path

    @property
    def enabled(self) -> bool:
        """False when no link is configured, which skips the artifact."""
        return bool(self.url.strip())

    @property
    def path(self) -> Path:
        """Where the file lands."""
        return self.target_dir / filename_from_url(self.url)


def registry(settings: InitSettings) -> list[Artifact]:
    """Describe every artifact the bootstrap knows how to fetch."""
    return [
        Artifact(name=MODEL, url=settings.model_url, sha256=settings.model_sha256, target_dir=settings.weights_dir),
        Artifact(
            name=CATBOOST, url=settings.catboost_url, sha256=settings.catboost_sha256, target_dir=settings.weights_dir
        ),
    ]


async def fetch_all(settings: InitSettings) -> None:
    """Download whatever is configured and not already on disk.

    Raises:
        StorageError: a download failed or its checksum did not match.
    """
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
        for artifact in registry(settings):
            if not artifact.enabled:
                logger.warning("No INIT_%s_URL: skipping the %s artifact", artifact.name.upper(), artifact.name)
                continue
            await fetch(
                url=artifact.url,
                destination=artifact.path,
                expected=artifact.sha256,
                force=settings.force,
                headers=_auth_headers(artifact.url, settings),
                client=client,
            )


async def model_version(settings: InitSettings) -> str:
    """The sha256 of the served weights: the version every gallery point must carry.

    The worker computes the same digest of the file it loads, so the two agree without
    either reading the other's configuration.

    Raises:
        ValidationError: no model is configured, or its file is missing.
    """
    model = registry(settings)[0]
    if not model.enabled:
        msg = "INIT_MODEL_URL is empty: the gallery cannot be embedded without the model"
        raise ValidationError(msg)
    if model.sha256.strip():
        return model.sha256.strip().lower()
    if not model.path.is_file():
        msg = f"the model is missing at {model.path}"
        raise ValidationError(msg)
    return await anyio.to_thread.run_sync(_sha256, model.path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _auth_headers(url: str, settings: InitSettings) -> dict[str, str]:
    """Attach the HF token, but only to Hugging Face; httpx drops it on the redirect to the CDN."""
    token = settings.hf_token.get_secret_value().strip()
    if not token or urlparse(url).hostname not in HF_HOSTS:
        return {}
    return {"Authorization": f"Bearer {token}"}

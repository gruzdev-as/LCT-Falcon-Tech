import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from init.src.configs.constants import DOWNLOAD_TIMEOUT_S, IMAGES_DIR
from init.src.configs.settings import InitSettings
from init.src.fetch import fetch, filename_from_url, unpack

logger = logging.getLogger(__name__)

MODEL = "model"
IMAGES = "images"
VECTORS = "vectors"


@dataclass(frozen=True, slots=True)
class Artifact:
    """One downloadable input of the bootstrap."""

    name: str
    url: str
    sha256: str
    target_dir: Path
    extract: bool

    @property
    def enabled(self) -> bool:
        """False when no link is configured, which skips the artifact."""
        return bool(self.url.strip())


def registry(settings: InitSettings) -> list[Artifact]:
    """Describe every artifact the bootstrap knows how to fetch."""
    return [
        Artifact(
            name=MODEL,
            url=settings.model_url,
            sha256=settings.model_sha256,
            target_dir=settings.weights_dir,
            extract=False,
        ),
        Artifact(
            name=IMAGES,
            url=settings.images_url,
            sha256=settings.images_sha256,
            target_dir=settings.artifacts_dir / IMAGES_DIR,
            extract=True,
        ),
        Artifact(
            name=VECTORS,
            url=settings.vectors_url,
            sha256=settings.vectors_sha256,
            target_dir=settings.artifacts_dir,
            extract=True,
        ),
    ]


async def fetch_all(settings: InitSettings) -> None:
    """Download whatever is configured and not already on disk.

    Raises:
        StorageError: a download failed or an archive could not be unpacked.
    """
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
        for artifact in registry(settings):
            if not artifact.enabled:
                logger.warning("No INIT_%s_URL: skipping the %s artifact", artifact.name.upper(), artifact.name)
                continue
            await _fetch_one(client, artifact, settings)


async def _fetch_one(client: httpx.AsyncClient, artifact: Artifact, settings: InitSettings) -> None:
    if not artifact.extract:
        await fetch(
            url=artifact.url,
            destination=artifact.target_dir / filename_from_url(artifact.url),
            expected=artifact.sha256,
            force=settings.force,
            client=client,
        )
        return

    # For an archive the unpacked directory is what matters, so it is the gate —
    # the download is skipped along with it, and the archive is not kept around.
    if _looks_unpacked(artifact.target_dir) and not settings.force:
        logger.info("Already unpacked, not downloading: %s", artifact.name)
        return

    archive = settings.artifacts_dir / ".downloads" / filename_from_url(artifact.url)
    await fetch(url=artifact.url, destination=archive, expected=artifact.sha256, client=client)
    await unpack(archive, artifact.target_dir)
    archive.unlink(missing_ok=True)


def _looks_unpacked(target: Path) -> bool:
    """Cheap check that a previous run already extracted something here."""
    return target.is_dir() and any(target.iterdir())

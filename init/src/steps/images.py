import logging
import mimetypes
from pathlib import Path

import anyio

from common.src.configs.constants import GALLERY_PREFIX
from common.src.storage import client as storage
from init.src.bundle import Bundle, GalleryEntry

logger = logging.getLogger(__name__)

_DEFAULT_CONTENT_TYPE = "image/jpeg"


def object_key(entry: GalleryEntry) -> str:
    """Object key a gallery image is stored under."""
    return f"{GALLERY_PREFIX}/{entry.image_path.lstrip('/')}"


async def upload_gallery(bundle: Bundle, *, concurrency: int, force: bool = False) -> int:
    """Put every gallery original into object storage, skipping what is already there.

    Args:
        bundle: validated bundle; its manifest decides what to upload.
        concurrency: how many uploads run at once.
        force: upload everything, even objects the bucket already holds.

    Returns:
        How many objects were uploaded by this call.

    Raises:
        StorageError: object storage is unreachable or an upload failed.
    """
    await storage.ensure_bucket()

    present: set[str] = set() if force else await storage.list_keys(f"{GALLERY_PREFIX}/")
    pending = [entry for entry in bundle.entries if object_key(entry) not in present]
    if not pending:
        logger.info("Gallery images already in object storage (%d objects), skipping upload", len(bundle.entries))
        return 0

    logger.info("Uploading %d of %d gallery images", len(pending), len(bundle.entries))
    limiter = anyio.CapacityLimiter(concurrency)
    async with anyio.create_task_group() as group:
        for entry in pending:
            group.start_soon(_upload_one, entry, bundle.images_dir, limiter)
    logger.info("Uploaded %d gallery images", len(pending))
    return len(pending)


async def _upload_one(entry: GalleryEntry, images_dir: Path, limiter: anyio.CapacityLimiter) -> None:
    async with limiter:
        path = entry.local_path(images_dir)
        data = await anyio.Path(path).read_bytes()
        await storage.save(object_key(entry), data, content_type=_content_type(path))


def _content_type(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed if guessed and guessed.startswith("image/") else _DEFAULT_CONTENT_TYPE

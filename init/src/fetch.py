import hashlib
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from common.src.exceptions import StorageError
from init.src.configs.constants import DOWNLOAD_CHUNK_BYTES, DOWNLOAD_TIMEOUT_S, PART_SUFFIX

logger = logging.getLogger(__name__)


def filename_from_url(url: str) -> str:
    """Take the filename an URL points at, falling back to a neutral name."""
    name = Path(unquote(urlparse(url).path)).name
    return name or "artifact"


async def fetch(
    url: str,
    destination: Path,
    *,
    expected: str = "",
    force: bool = False,
    headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Download one file unless it is already here.

    Args:
        url: what to download.
        destination: final path of the file.
        expected: sha256 the download must match; empty skips verification.
        force: download even when the file is already there.
        headers: extra request headers, e.g. authorization.
        client: HTTP client to reuse; one is created when omitted.

    Returns:
        Whether anything was actually downloaded.

    Raises:
        StorageError: the transfer failed or the checksum did not match.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and not force:
        logger.info("Already present, not downloading: %s", destination.name)
        return False

    part = destination.with_name(destination.name + PART_SUFFIX)
    part.unlink(missing_ok=True)

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True)
    try:
        digest = await _stream_to_file(client, url, part, headers)
    finally:
        if owns_client:
            await client.aclose()

    if expected and digest != expected.strip().lower():
        part.unlink(missing_ok=True)
        msg = f"checksum mismatch for {url}"
        raise StorageError(msg, details={"expected": expected, "actual": digest})

    part.replace(destination)
    logger.info("Downloaded %s (sha256 %s)", destination.name, digest[:12])
    return True


async def _stream_to_file(client: httpx.AsyncClient, url: str, part: Path, headers: dict[str, str] | None) -> str:
    """Stream a response into a file, digesting it on the way."""
    digest = hashlib.sha256()
    try:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
                    digest.update(chunk)
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        part.unlink(missing_ok=True)
        msg = f"failed to download {url}"
        raise StorageError(msg) from exc
    return digest.hexdigest()

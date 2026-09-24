import hashlib
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import anyio.to_thread
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
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Download one file unless it is already here.

    Args:
        url: what to download.
        destination: final path of the file.
        expected: sha256 the download must match; empty skips verification.
        force: download even when the file is already there.
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
        digest = await _stream_to_file(client, url, part)
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


async def _stream_to_file(client: httpx.AsyncClient, url: str, part: Path) -> str:
    """Stream a response into a file, digesting it on the way."""
    digest = hashlib.sha256()
    try:
        async with client.stream("GET", url) as response:
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


async def unpack(archive: Path, destination: Path) -> None:
    """Extract an archive into a directory, refusing entries that escape it.

    Raises:
        StorageError: the file is not a zip or tar this interpreter can read, or extraction failed.
    """
    await anyio.to_thread.run_sync(_unpack_sync, archive, destination)


def _unpack_sync(archive: Path, destination: Path) -> None:
    is_zip = zipfile.is_zipfile(archive)
    if not is_zip and not tarfile.is_tarfile(archive):
        msg = f"{archive.name} is not a zip or a tar this Python can read"
        # tarfile handles gz, bz2 and xz; zstd needs Python 3.14 or an extra library.
        raise StorageError(msg, details={"supported": "zip, tar, tar.gz, tar.bz2, tar.xz"})

    destination.mkdir(parents=True, exist_ok=True)
    try:
        if is_zip:
            with zipfile.ZipFile(archive) as bundle:
                _check_members(bundle.namelist(), destination)
                bundle.extractall(destination)  # noqa: S202  # members checked above
        else:
            with tarfile.open(archive) as bundle:
                _check_members(bundle.getnames(), destination)
                bundle.extractall(destination, filter="data")
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        msg = f"failed to unpack {archive.name}"
        raise StorageError(msg) from exc
    finally:
        shutil.rmtree(destination / "__MACOSX", ignore_errors=True)


def _check_members(names: list[str], destination: Path) -> None:
    """Reject absolute paths and ``..`` traversal before anything is written."""
    root = destination.resolve()
    for name in names:
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            msg = f"archive entry escapes the destination: {name}"
            raise StorageError(msg)

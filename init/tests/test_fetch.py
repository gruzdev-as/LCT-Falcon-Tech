import hashlib
import tarfile
from pathlib import Path

import httpx
import pytest

from common.src.exceptions import StorageError
from init.src.configs.constants import PART_SUFFIX
from init.src.fetch import fetch, unpack

PAYLOAD = b"gallery-artifact-bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.test/artifact.bin"


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def serve(body: bytes = PAYLOAD):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return handler


def refuse(_request: httpx.Request) -> httpx.Response:
    raise AssertionError("must not hit the network")


async def test_downloads_and_verifies(tmp_path: Path) -> None:
    async with client_for(serve()) as client:
        downloaded = await fetch(URL, tmp_path / "artifact.bin", expected=DIGEST, client=client)

    assert downloaded
    assert (tmp_path / "artifact.bin").read_bytes() == PAYLOAD


async def test_downloads_without_a_checksum(tmp_path: Path) -> None:
    async with client_for(serve()) as client:
        assert await fetch(URL, tmp_path / "artifact.bin", client=client)


async def test_skips_what_is_already_here(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(PAYLOAD)

    async with client_for(refuse) as client:
        assert not await fetch(URL, destination, expected=DIGEST, client=client)


async def test_force_downloads_anyway(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"stale")

    async with client_for(serve()) as client:
        assert await fetch(URL, destination, force=True, client=client)

    assert destination.read_bytes() == PAYLOAD


async def test_a_mismatch_leaves_nothing_behind(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"

    async with client_for(serve(b"tampered")) as client:
        with pytest.raises(StorageError, match="checksum mismatch"):
            await fetch(URL, destination, expected=DIGEST, client=client)

    assert not destination.exists()
    assert not destination.with_name(destination.name + PART_SUFFIX).exists()


async def test_an_interrupted_transfer_does_not_look_complete(tmp_path: Path) -> None:
    """A truncated download must not pass the next run's `file exists` check."""

    def broken(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection dropped")

    destination = tmp_path / "artifact.bin"
    async with client_for(broken) as client:
        with pytest.raises(StorageError, match="failed to download"):
            await fetch(URL, destination, client=client)

    assert not destination.exists()
    assert not destination.with_name(destination.name + PART_SUFFIX).exists()


async def test_unpack_detects_the_format_by_content_not_extension(tmp_path: Path) -> None:
    """Publishers name archives whatever they like; the bytes decide."""
    source = tmp_path / "payload.txt"
    source.write_text("hello")
    archive = tmp_path / "gallery.bin"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="payload.txt")

    await unpack(archive, tmp_path / "out")

    assert (tmp_path / "out" / "payload.txt").read_text() == "hello"


async def test_unpack_names_the_problem_for_an_unreadable_format(tmp_path: Path) -> None:
    """A .tar.zst would sail past an extension check and fail deep inside tarfile."""
    archive = tmp_path / "gallery.tar.zst"
    archive.write_bytes(b"\x28\xb5\x2f\xfd not really zstd either")

    with pytest.raises(StorageError, match="not a zip or a tar"):
        await unpack(archive, tmp_path / "out")


async def test_unpack_refuses_an_entry_that_escapes(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    victim = tmp_path / "payload"
    victim.write_bytes(b"x")
    with tarfile.open(archive, "w") as bundle:
        bundle.add(victim, arcname="../escaped")

    with pytest.raises(StorageError, match="escapes the destination"):
        await unpack(archive, tmp_path / "out")

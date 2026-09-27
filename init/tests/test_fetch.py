import hashlib
from pathlib import Path

import httpx
import pytest

from common.src.exceptions import StorageError
from init.src.configs.constants import PART_SUFFIX
from init.src.fetch import fetch

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

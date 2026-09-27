import hashlib
from pathlib import Path

import httpx
import pytest

from common.src.exceptions import ValidationError
from init.src import artifacts as artifacts_module
from init.src.artifacts import CATBOOST, MODEL, fetch_all, model_version, registry
from init.src.configs.settings import InitSettings

WEIGHTS = b"pretend-these-are-weights"


def settings_for(tmp_path: Path, **overrides) -> InitSettings:
    return InitSettings(gallery_dir=tmp_path / "gallery", weights_dir=tmp_path / "weights", **overrides)


def test_registry_lists_every_artifact(tmp_path: Path) -> None:
    assert [artifact.name for artifact in registry(settings_for(tmp_path))] == [MODEL, CATBOOST]


def test_an_artifact_without_a_link_is_disabled(tmp_path: Path) -> None:
    artifacts = {item.name: item for item in registry(settings_for(tmp_path))}

    assert not artifacts[MODEL].enabled


async def test_missing_links_are_skipped_not_fatal(tmp_path: Path) -> None:
    """Nothing is published yet, so an empty URL must still let the stack come up."""
    await fetch_all(settings_for(tmp_path))


async def test_fetching_weights_lands_them_in_the_weights_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = _serve_bytes(monkeypatch, WEIGHTS)

    await fetch_all(settings_for(tmp_path, model_url="https://example.test/eva02.pt"))

    assert (tmp_path / "weights" / "eva02.pt").read_bytes() == WEIGHTS
    assert served["hits"] == 1


async def test_the_catboost_head_lands_next_to_the_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_bytes(monkeypatch, WEIGHTS)

    await fetch_all(settings_for(tmp_path, catboost_url="https://example.test/eva02_catboost.cbm"))

    assert (tmp_path / "weights" / "eva02_catboost.cbm").read_bytes() == WEIGHTS


async def test_the_hf_token_goes_only_to_hugging_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token for a gated repo must not leak to whoever hosts the other artifacts."""
    served = _serve_bytes(monkeypatch, WEIGHTS)

    await fetch_all(
        settings_for(
            tmp_path,
            hf_token="hf_secret",  # noqa: S106  # a fake token for the test
            model_url="https://huggingface.co/org/repo/resolve/main/eva02.pt",
            catboost_url="https://example.test/eva02_catboost.cbm",
        )
    )

    assert served["auth"] == ["Bearer hf_secret", None]


async def test_a_second_run_downloads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served = _serve_bytes(monkeypatch, WEIGHTS)
    settings = settings_for(tmp_path, model_url="https://example.test/eva02.pt")

    await fetch_all(settings)
    await fetch_all(settings)

    assert served["hits"] == 1


async def test_force_downloads_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served = _serve_bytes(monkeypatch, WEIGHTS)

    await fetch_all(settings_for(tmp_path, model_url="https://example.test/eva02.pt"))
    await fetch_all(settings_for(tmp_path, model_url="https://example.test/eva02.pt", force=True))

    assert served["hits"] == 2


async def test_the_model_version_is_the_configured_checksum(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, model_url="https://example.test/eva02.pt", model_sha256="ABC123")
    assert await model_version(settings) == "abc123"


async def test_without_a_checksum_the_model_version_is_the_file_digest(tmp_path: Path) -> None:
    """The worker hashes the file it loads, so both sides must land on the same digest."""
    settings = settings_for(tmp_path, model_url="https://example.test/eva02.pt")
    settings.weights_dir.mkdir(parents=True)
    (settings.weights_dir / "eva02.pt").write_bytes(WEIGHTS)

    assert await model_version(settings) == hashlib.sha256(WEIGHTS).hexdigest()


async def test_a_gallery_cannot_be_built_without_a_model(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="INIT_MODEL_URL"):
        await model_version(settings_for(tmp_path))


async def test_a_missing_model_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="missing"):
        await model_version(settings_for(tmp_path, model_url="https://example.test/eva02.pt"))


def _serve_bytes(monkeypatch: pytest.MonkeyPatch, body: bytes) -> dict:
    """Replace the HTTP transport with one serving fixed bytes; counts downloads."""
    counter = {"hits": 0, "auth": []}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["hits"] += 1
        counter["auth"].append(request.headers.get("authorization"))
        return httpx.Response(200, content=body)

    original = httpx.AsyncClient

    def build(*_args, **kwargs) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(**kwargs)

    monkeypatch.setattr(artifacts_module.httpx, "AsyncClient", build)
    return counter

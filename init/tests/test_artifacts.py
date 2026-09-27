import tarfile
from pathlib import Path

import httpx
import pytest

from init.src import artifacts as artifacts_module
from init.src.artifacts import CATBOOST, IMAGES, MODEL, VECTORS, fetch_all, registry
from init.src.configs.settings import InitSettings

WEIGHTS = b"pretend-these-are-weights"


def settings_for(tmp_path: Path, **overrides) -> InitSettings:
    return InitSettings(artifacts_dir=tmp_path / "gallery", weights_dir=tmp_path / "weights", **overrides)


def test_registry_lists_every_artifact(tmp_path: Path) -> None:
    assert [artifact.name for artifact in registry(settings_for(tmp_path))] == [MODEL, CATBOOST, IMAGES, VECTORS]


def test_an_artifact_without_a_link_is_disabled(tmp_path: Path) -> None:
    artifacts = {item.name: item for item in registry(settings_for(tmp_path))}

    assert not artifacts[MODEL].enabled


def test_weights_are_kept_as_a_file_and_the_rest_unpacked(tmp_path: Path) -> None:
    artifacts = {item.name: item for item in registry(settings_for(tmp_path))}

    assert not artifacts[MODEL].extract
    assert not artifacts[CATBOOST].extract
    assert artifacts[IMAGES].extract
    assert artifacts[VECTORS].extract


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


async def test_an_archive_artifact_is_unpacked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _make_archive(tmp_path / "src", tmp_path / "vectors.tar")
    _serve_bytes(monkeypatch, archive.read_bytes())

    settings = settings_for(tmp_path, vectors_url="https://example.test/vectors.tar")
    await fetch_all(settings)

    assert (settings.artifacts_dir / "bundle.json").is_file()


async def test_the_downloaded_archive_is_not_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeping it would cost a second copy of the gallery on disk for nothing."""
    archive = _make_archive(tmp_path / "src", tmp_path / "vectors.tar")
    _serve_bytes(monkeypatch, archive.read_bytes())

    settings = settings_for(tmp_path, vectors_url="https://example.test/vectors.tar")
    await fetch_all(settings)

    assert not list((settings.artifacts_dir / ".downloads").glob("*.tar"))


async def test_an_unpacked_archive_is_not_downloaded_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _make_archive(tmp_path / "src", tmp_path / "vectors.tar")
    served = _serve_bytes(monkeypatch, archive.read_bytes())
    settings = settings_for(tmp_path, vectors_url="https://example.test/vectors.tar")

    await fetch_all(settings)
    await fetch_all(settings)

    assert served["hits"] == 1


async def test_force_refetches_and_unpacks_an_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _make_archive(tmp_path / "src", tmp_path / "vectors.tar")
    served = _serve_bytes(monkeypatch, archive.read_bytes())

    await fetch_all(settings_for(tmp_path, vectors_url="https://example.test/vectors.tar"))
    await fetch_all(settings_for(tmp_path, vectors_url="https://example.test/vectors.tar", force=True))

    assert served["hits"] == 2


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


def _make_archive(source: Path, archive: Path) -> Path:
    source.mkdir(parents=True, exist_ok=True)
    (source / "bundle.json").write_text("{}")
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source / "bundle.json", arcname="bundle.json")
    return archive

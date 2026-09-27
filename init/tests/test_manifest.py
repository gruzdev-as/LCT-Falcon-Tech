from pathlib import Path

import pytest

from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError
from init.src.configs.constants import MANIFEST_COLUMNS, MANIFEST_FILE
from init.src.gallery import load_gallery
from init.tests.helpers import write_gallery


def test_loads_a_valid_gallery(tmp_path: Path) -> None:
    gallery = load_gallery(write_gallery(tmp_path, count=4))

    assert gallery.count == 4
    assert gallery.entries[0].image_id == "img-0"
    assert gallery.entries[0].bbox == BBox(x=1, y=2, width=30, height=20)
    assert gallery.entries[0].local_path(gallery.images_dir).is_file()


def test_ids_are_carried_as_strings(tmp_path: Path) -> None:
    """Candidate types them as `str | None`, and pydantic v2 will not coerce an int."""
    entry = load_gallery(write_gallery(tmp_path)).entries[1]

    assert entry.vehicle_id == "0"
    assert entry.camera_id == "cam-1"


def test_the_version_follows_the_manifest(tmp_path: Path) -> None:
    first = load_gallery(write_gallery(tmp_path / "a", count=3)).version
    same = load_gallery(write_gallery(tmp_path / "b", count=3)).version
    other = load_gallery(write_gallery(tmp_path / "c", count=4)).version

    assert first == same
    assert first != other


def test_a_missing_gallery_says_what_to_mount(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="mount a directory"):
        load_gallery(tmp_path)


def test_rejects_a_manifest_row_without_an_image(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not in the gallery"):
        load_gallery(write_gallery(tmp_path, skip_image=1))


def test_rejects_duplicate_image_ids(tmp_path: Path) -> None:
    root = write_gallery(tmp_path, count=2)
    manifest = root / MANIFEST_FILE
    manifest.write_text(manifest.read_text().replace("img-1", "img-0"))

    with pytest.raises(ValidationError, match="repeats image ids"):
        load_gallery(root)


@pytest.mark.parametrize("column", ["image_id", "bbox_width"])
def test_rejects_a_manifest_missing_a_column(tmp_path: Path, column: str) -> None:
    """The box is required: every gallery image comes with its vehicle marked."""
    columns = tuple(name for name in MANIFEST_COLUMNS if name != column)

    with pytest.raises(ValidationError, match="missing required columns"):
        load_gallery(write_gallery(tmp_path, columns=columns))


@pytest.mark.parametrize("box", [{"bbox_height": ""}, {"bbox_x": "left"}, {"bbox_width": 3}])
def test_rejects_a_row_without_a_valid_box(tmp_path: Path, box: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="without a valid bbox"):
        load_gallery(write_gallery(tmp_path, box=box))


def test_rejects_an_empty_manifest(tmp_path: Path) -> None:
    root = write_gallery(tmp_path, count=1)
    (root / MANIFEST_FILE).write_text(",".join(MANIFEST_COLUMNS) + "\n")

    with pytest.raises(ValidationError, match="no rows"):
        load_gallery(root)

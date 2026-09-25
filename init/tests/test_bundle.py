import json
from pathlib import Path

import numpy as np
import pytest

from common.src.exceptions import ValidationError
from init.src.bundle import bundle_present, load_bundle
from init.src.configs.constants import BUNDLE_FILE, EMBEDDINGS_FILE, MANIFEST_FILE
from init.tests.helpers import DIM, MODEL, VERSION, write_bundle


def test_loads_a_valid_bundle(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=5)

    bundle = load_bundle(tmp_path)

    assert bundle.count == 5
    assert bundle.embedding_dim == DIM
    assert bundle.version == VERSION
    assert bundle.model_name == MODEL
    assert bundle.embeddings.dtype == np.float32
    assert bundle.entries[0].vehicle_id == "0"


def test_reports_a_missing_bundle(tmp_path: Path) -> None:
    assert not bundle_present(tmp_path)
    write_bundle(tmp_path)
    assert bundle_present(tmp_path)


def test_rejects_a_manifest_longer_than_the_matrix(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=4)
    np.save(tmp_path / EMBEDDINGS_FILE, np.eye(3, DIM, dtype=np.float32))

    with pytest.raises(ValidationError, match="disagree on how many"):
        load_bundle(tmp_path)


def test_rejects_a_declared_dimension_that_does_not_match(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    meta = json.loads((tmp_path / BUNDLE_FILE).read_text())
    meta["embedding_dim"] = DIM + 1
    (tmp_path / BUNDLE_FILE).write_text(json.dumps(meta))

    with pytest.raises(ValidationError, match="different embedding dimension"):
        load_bundle(tmp_path)


def test_rejects_a_declared_count_that_does_not_match(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=4, declared_count=9)

    with pytest.raises(ValidationError, match="different image count"):
        load_bundle(tmp_path)


def test_rejects_unnormalized_embeddings(tmp_path: Path) -> None:
    write_bundle(tmp_path, normalize=False)

    with pytest.raises(ValidationError, match="not L2-normalized"):
        load_bundle(tmp_path)


def test_rejects_a_manifest_row_without_an_image(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=3, skip_image=1)

    with pytest.raises(ValidationError, match="does not contain"):
        load_bundle(tmp_path)


def test_rejects_duplicate_image_ids(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=2)
    path = tmp_path / MANIFEST_FILE
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[1], lines[1]]) + "\n")

    with pytest.raises(ValidationError, match="repeats image ids"):
        load_bundle(tmp_path)


def test_rejects_a_manifest_missing_a_column(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=2)
    path = tmp_path / MANIFEST_FILE
    rows = [line.rsplit(",", 1)[0] for line in path.read_text().splitlines()]
    path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValidationError, match="missing required columns"):
        load_bundle(tmp_path)


def test_carries_the_model_that_produced_the_vectors(tmp_path: Path) -> None:
    """Which model made the gallery has to be visible: the demo is unusable if it drifts."""
    bundle = load_bundle(write_bundle(tmp_path, model_name="model-b"))

    assert bundle.model_name == "model-b"


def test_reads_the_optional_bbox_columns(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=3, boxes=True)

    entries = load_bundle(tmp_path).entries

    assert entries[0].bbox is not None
    assert (entries[0].bbox.x, entries[0].bbox.y) == (1.0, 2.0)
    assert (entries[2].bbox.x, entries[2].bbox.width) == (3.0, 30.0)


def test_a_manifest_without_bbox_columns_still_loads(tmp_path: Path) -> None:
    """The columns are optional on purpose: bundles built before them must keep working."""
    write_bundle(tmp_path, count=2)

    assert [entry.bbox for entry in load_bundle(tmp_path).entries] == [None, None]


def test_rejects_a_half_filled_bbox(tmp_path: Path) -> None:
    write_bundle(tmp_path, count=2, partial_box=True)

    with pytest.raises(ValidationError, match="partial bbox"):
        load_bundle(tmp_path)

import csv
import json
from pathlib import Path

import numpy as np

from inference.tests.helpers import make_image
from init.src.configs.constants import (
    BBOX_COLUMNS,
    BUNDLE_FILE,
    EMBEDDINGS_FILE,
    IMAGES_DIR,
    MANIFEST_COLUMNS,
    MANIFEST_FILE,
)

DIM = 8
VERSION = "test-bundle-1"
MODEL = "test-model"


def write_bundle(
    root: Path,
    *,
    count: int = 4,
    dim: int = DIM,
    version: str = VERSION,
    model_name: str = MODEL,
    normalize: bool = True,
    declared_count: int | None = None,
    skip_image: int | None = None,
    boxes: bool = False,
    partial_box: bool = False,
) -> Path:
    """Lay a valid bundle out on disk, with hooks for the broken variants.

    Images come from the existing `make_image` helper rather than a generator of
    our own, so the two suites produce the same kind of fixture.
    """
    images_dir = root / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(count):
        name = f"car_{index}.png"
        if index != skip_image:
            (images_dir / name).write_bytes(make_image(index))
        row = {
            "image_id": f"img-{index}",
            "image_path": name,
            "vehicle_id": f"{index // 2}",
            "camera_id": f"cam-{index % 2}",
        }
        if boxes or partial_box:
            row |= {"bbox_x": 1 + index, "bbox_y": 2 + index, "bbox_width": 30, "bbox_height": 20}
        if partial_box:
            row["bbox_height"] = ""
        rows.append(row)

    columns = list(MANIFEST_COLUMNS) + (list(BBOX_COLUMNS) if boxes or partial_box else [])
    with (root / MANIFEST_FILE).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    vectors = np.random.default_rng(0).normal(size=(count, dim)).astype(np.float32)
    if normalize:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    else:
        vectors *= 5.0
    np.save(root / EMBEDDINGS_FILE, vectors)

    (root / BUNDLE_FILE).write_text(
        json.dumps({
            "bundle_version": version,
            "model_name": model_name,
            "embedding_dim": dim,
            "distance": "cosine",
            "normalized": normalize,
            "count": count if declared_count is None else declared_count,
        })
    )
    return root

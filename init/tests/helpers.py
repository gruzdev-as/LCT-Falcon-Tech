import csv
import json
from pathlib import Path

import numpy as np

from inference.tests.helpers import make_image
from init.src.configs.constants import BUNDLE_FILE, EMBEDDINGS_FILE, IMAGES_DIR, MANIFEST_COLUMNS, MANIFEST_FILE

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
        rows.append({
            "image_id": f"img-{index}",
            "image_path": name,
            "vehicle_id": f"{index // 2}",
            "camera_id": f"cam-{index % 2}",
        })

    with (root / MANIFEST_FILE).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
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

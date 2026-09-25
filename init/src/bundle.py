import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError
from init.src.configs.constants import (
    BBOX_COLUMNS,
    BUNDLE_FILE,
    EMBEDDINGS_FILE,
    IMAGES_DIR,
    MANIFEST_COLUMNS,
    MANIFEST_FILE,
)

logger = logging.getLogger(__name__)

_NORMALIZED_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """One row of the manifest, paired with its image on disk."""

    image_id: str
    image_path: str
    """Path relative to the bundle's ``images/`` directory."""

    vehicle_id: str | None
    camera_id: str | None

    bbox: BBox | None = None
    """The vehicle inside the frame. None means the frame is already the vehicle."""

    def local_path(self, images_dir: Path) -> Path:
        """Where the original lives inside the artifacts directory."""
        return images_dir / self.image_path


@dataclass(frozen=True, slots=True)
class Bundle:
    """A validated set of gallery embeddings and the metadata describing them."""

    version: str
    model_name: str
    embedding_dim: int
    distance: str
    entries: list[GalleryEntry]
    embeddings: np.ndarray
    """float32 of shape ``(len(entries), embedding_dim)``, L2-normalized."""

    images_dir: Path

    @property
    def count(self) -> int:
        """How many gallery images the bundle carries."""
        return len(self.entries)


def bundle_present(artifacts_dir: Path) -> bool:
    """Report whether an artifacts directory holds a bundle at all."""
    return (artifacts_dir / BUNDLE_FILE).is_file()


def load_bundle(artifacts_dir: Path) -> Bundle:
    """Read and validate the bundle in an artifacts directory.

    Args:
        artifacts_dir: directory the vectors artifact was unpacked into.

    Returns:
        The validated bundle.

    Raises:
        ValidationError: any part of the bundle is missing or inconsistent.
    """
    meta = _read_meta(artifacts_dir / BUNDLE_FILE)
    entries = _read_manifest(artifacts_dir / MANIFEST_FILE)
    embeddings = _read_embeddings(artifacts_dir / EMBEDDINGS_FILE)

    if len(entries) != len(embeddings):
        msg = "manifest and embeddings disagree on how many gallery images there are"
        raise ValidationError(msg, details={"manifest": len(entries), "embeddings": len(embeddings)})

    declared_dim = int(meta["embedding_dim"])
    if embeddings.shape[1] != declared_dim:
        msg = "bundle.json declares a different embedding dimension than embeddings.npy holds"
        raise ValidationError(msg, details={"declared": declared_dim, "actual": int(embeddings.shape[1])})

    declared_count = meta.get("count")
    if declared_count is not None and int(declared_count) != len(entries):
        msg = "bundle.json declares a different image count than the manifest holds"
        raise ValidationError(msg, details={"declared": int(declared_count), "actual": len(entries)})

    _check_normalized(embeddings)

    images_dir = artifacts_dir / IMAGES_DIR
    _check_images_present(entries, images_dir)

    return Bundle(
        version=str(meta["bundle_version"]),
        model_name=str(meta["model_name"]),
        embedding_dim=declared_dim,
        distance=str(meta.get("distance", "cosine")),
        entries=entries,
        embeddings=embeddings,
        images_dir=images_dir,
    )


def _read_meta(path: Path) -> dict:
    try:
        meta = json.loads(path.read_text())
    except FileNotFoundError as exc:
        msg = f"{BUNDLE_FILE} is missing from the artifacts directory"
        raise ValidationError(msg, details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        msg = f"{BUNDLE_FILE} is not valid JSON"
        raise ValidationError(msg, details={"path": str(path)}) from exc

    missing = [key for key in ("bundle_version", "model_name", "embedding_dim") if key not in meta]
    if missing:
        msg = f"{BUNDLE_FILE} is missing required keys"
        raise ValidationError(msg, details={"missing": missing})
    return meta


def _read_manifest(path: Path) -> list[GalleryEntry]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        msg = f"{MANIFEST_FILE} is missing from the artifacts directory"
        raise ValidationError(msg, details={"path": str(path)}) from exc

    if not rows:
        msg = f"{MANIFEST_FILE} has no rows"
        raise ValidationError(msg)

    missing = [column for column in MANIFEST_COLUMNS if column not in rows[0]]
    if missing:
        msg = f"{MANIFEST_FILE} is missing required columns"
        raise ValidationError(msg, details={"missing": missing, "found": sorted(rows[0])})

    entries = [
        GalleryEntry(
            image_id=str(row["image_id"]).strip(),
            image_path=str(row["image_path"]).strip(),
            vehicle_id=_optional(row["vehicle_id"]),
            camera_id=_optional(row["camera_id"]),
            bbox=_read_bbox(row),
        )
        for row in rows
    ]

    duplicates = len(entries) - len({entry.image_id for entry in entries})
    if duplicates:
        msg = f"{MANIFEST_FILE} repeats image ids"
        raise ValidationError(msg, details={"duplicates": duplicates})
    return entries


def _optional(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def _read_bbox(row: dict[str, str | None]) -> BBox | None:
    """Read the optional bbox columns, refusing a half-filled box.

    Raises:
        ValidationError: some of the four columns are set and the rest are not.
    """
    values = [_optional(row.get(column)) for column in BBOX_COLUMNS]
    if not any(values):
        return None
    if not all(values):
        msg = f"{MANIFEST_FILE} has a partial bbox"
        raise ValidationError(msg, details={"image_id": row.get("image_id"), "columns": list(BBOX_COLUMNS)})

    x, y, width, height = (float(value) for value in values)  # type: ignore[arg-type]
    return BBox(x=x, y=y, width=width, height=height)


def _read_embeddings(path: Path) -> np.ndarray:
    try:
        matrix = np.load(path)
    except FileNotFoundError as exc:
        msg = f"{EMBEDDINGS_FILE} is missing from the artifacts directory"
        raise ValidationError(msg, details={"path": str(path)}) from exc
    except ValueError as exc:
        msg = f"{EMBEDDINGS_FILE} is not a readable .npy array"
        raise ValidationError(msg, details={"path": str(path)}) from exc

    if matrix.ndim != 2:
        msg = f"{EMBEDDINGS_FILE} must be a 2D matrix"
        raise ValidationError(msg, details={"shape": list(matrix.shape)})
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _check_normalized(embeddings: np.ndarray) -> None:
    """Refuse vectors that are not unit length."""
    norms = np.linalg.norm(embeddings, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > _NORMALIZED_TOLERANCE:
        msg = "embeddings are not L2-normalized"
        raise ValidationError(msg, details={"max_deviation": worst})


def _check_images_present(entries: list[GalleryEntry], images_dir: Path) -> None:
    missing = [entry.image_path for entry in entries if not entry.local_path(images_dir).is_file()]
    if missing:
        msg = "the manifest references images that the images artifact does not contain"
        raise ValidationError(msg, details={"missing_count": len(missing), "examples": missing[:5]})

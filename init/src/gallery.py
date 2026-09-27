import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError as SchemaError

from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError
from init.src.configs.constants import BBOX_COLUMNS, GALLERY_VERSION_CHARS, IMAGES_DIR, MANIFEST_COLUMNS, MANIFEST_FILE

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """One row of the manifest, paired with its image on disk."""

    image_id: str
    image_path: str
    """Path relative to the gallery's ``images/`` directory."""

    vehicle_id: str | None
    camera_id: str | None
    bbox: BBox
    """The vehicle inside the frame, in absolute pixels."""

    def local_path(self, images_dir: Path) -> Path:
        """Where the original lives in the mounted gallery."""
        return images_dir / self.image_path


@dataclass(frozen=True, slots=True)
class Gallery:
    """The mounted gallery: its images and the manifest describing them."""

    version: str
    """Fingerprint of the manifest; recorded with every metadata row."""

    entries: list[GalleryEntry]
    images_dir: Path

    @property
    def count(self) -> int:
        """How many gallery images there are."""
        return len(self.entries)


def load_gallery(gallery_dir: Path) -> Gallery:
    """Read and validate the gallery mounted at ``gallery_dir``.

    Raises:
        ValidationError: the manifest is missing or inconsistent, or an image is missing.
    """
    manifest = gallery_dir / MANIFEST_FILE
    try:
        raw = manifest.read_bytes()
    except FileNotFoundError as exc:
        msg = f"no gallery at {gallery_dir}: mount a directory with {MANIFEST_FILE} and {IMAGES_DIR}/"
        raise ValidationError(msg, details={"path": str(manifest)}) from exc

    entries = _read_manifest(raw.decode("utf-8-sig").splitlines())
    images_dir = gallery_dir / IMAGES_DIR
    _check_images_present(entries, images_dir)
    return Gallery(
        version=hashlib.sha256(raw).hexdigest()[:GALLERY_VERSION_CHARS],
        entries=entries,
        images_dir=images_dir,
    )


def _read_manifest(lines: list[str]) -> list[GalleryEntry]:
    rows = list(csv.DictReader(lines))
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


def _read_bbox(row: dict[str, str | None]) -> BBox:
    """Read the vehicle box, which every row must carry.

    Raises:
        ValidationError: a box column is empty or not a number, or the box is too small.
    """
    try:
        x, y, width, height = (float(str(row[column]).strip()) for column in BBOX_COLUMNS)
        return BBox(x=x, y=y, width=width, height=height)
    except (ValueError, SchemaError) as exc:
        msg = f"{MANIFEST_FILE} has a row without a valid bbox"
        raise ValidationError(msg, details={"image_id": row.get("image_id"), "error": str(exc)}) from exc


def _check_images_present(entries: list[GalleryEntry], images_dir: Path) -> None:
    missing = [entry.image_path for entry in entries if not entry.local_path(images_dir).is_file()]
    if missing:
        msg = "the manifest references images that are not in the gallery"
        raise ValidationError(msg, details={"missing_count": len(missing), "examples": missing[:5]})

import csv
from pathlib import Path

from inference.tests.helpers import make_image
from init.src.configs.constants import IMAGES_DIR, MANIFEST_COLUMNS, MANIFEST_FILE


def write_gallery(
    root: Path,
    *,
    count: int = 4,
    skip_image: int | None = None,
    columns: tuple[str, ...] = MANIFEST_COLUMNS,
    box: dict[str, object] | None = None,
) -> Path:
    """Lay a mounted gallery out on disk, with hooks for the broken variants.

    Images come from the existing `make_image` helper rather than a generator of
    our own, so the two suites produce the same kind of fixture.

    Args:
        root: the gallery directory.
        count: how many images.
        skip_image: index of a row whose image is left out.
        columns: manifest columns to write; drop one to break the header.
        box: overrides for the first row's bbox columns.
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
            "bbox_x": 1 + index,
            "bbox_y": 2 + index,
            "bbox_width": 30,
            "bbox_height": 20,
        })
    if box:
        rows[0] |= box

    with (root / MANIFEST_FILE).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return root

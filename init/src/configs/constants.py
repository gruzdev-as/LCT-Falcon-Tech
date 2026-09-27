from typing import Final

### Mounted gallery, relative to the gallery directory

MANIFEST_FILE: Final[str] = "manifest.csv"
IMAGES_DIR: Final[str] = "images"

# The box is required: every gallery image comes with its vehicle marked, the same way
# a search query does, and the model embeds the crop around it.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "image_id",
    "image_path",
    "vehicle_id",
    "camera_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
)
BBOX_COLUMNS: Final[tuple[str, ...]] = ("bbox_x", "bbox_y", "bbox_width", "bbox_height")

# How many characters of the manifest's sha256 name a gallery version
GALLERY_VERSION_CHARS: Final[int] = 16

### Downloading

PART_SUFFIX: Final[str] = ".part"
DOWNLOAD_CHUNK_BYTES: Final[int] = 1024 * 1024
DOWNLOAD_TIMEOUT_S: Final[float] = 600.0
HASH_CHUNK_BYTES: Final[int] = 8 * 1024 * 1024
# The HF token is sent only to these hosts, never to wherever the other links point.
HF_HOSTS: Final[frozenset[str]] = frozenset({"huggingface.co", "hf.co"})

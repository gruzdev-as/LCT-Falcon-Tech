import uuid
from typing import Final

### Artifact bundle layout, relative to the artifacts directory

BUNDLE_FILE: Final[str] = "bundle.json"
EMBEDDINGS_FILE: Final[str] = "embeddings.npy"
MANIFEST_FILE: Final[str] = "manifest.csv"
IMAGES_DIR: Final[str] = "images"

MANIFEST_COLUMNS: Final[tuple[str, ...]] = ("image_id", "image_path", "vehicle_id", "camera_id")

### Downloading

PART_SUFFIX: Final[str] = ".part"
DOWNLOAD_CHUNK_BYTES: Final[int] = 1024 * 1024
DOWNLOAD_TIMEOUT_S: Final[float] = 600.0

### Qdrant point ids (kinda weird TODO fix in the future if i have the time)

# Point ids must be uint64 or UUID, so image ids are mapped through uuid5.
# A one-off uuid4; the value is arbitrary but frozen - changing it remaps every id,
# and the next run indexes a second copy of the gallery instead of updating it.
POINT_NAMESPACE: Final[uuid.UUID] = uuid.UUID("4e3c7a67-36f1-4148-9b2e-e775a7990da8")

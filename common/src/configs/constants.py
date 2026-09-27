import os
from typing import Final

### Logging

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO").upper()

### Redis

TASK_STREAM: Final[str] = "falcon:tasks"
# Gallery images to embed. Separate from the search stream so a search never queues behind the gallery
GALLERY_STREAM: Final[str] = "falcon:gallery"
INFERENCE_GROUP: Final[str] = "inference-workers"
STREAM_PAYLOAD_FIELD: Final[str] = "payload"
STREAM_MAXLEN: Final[int] = 100_000

### Redis: task state and results

TASK_KEY: Final[str] = "task:{task_id}"
RESULT_KEY: Final[str] = "result:{task_id}"
TASK_TTL_SECONDS: Final[int] = 3600
RESULT_TTL_SECONDS: Final[int] = 3600

### Redis: gallery indexing

# Hash written by init: what the gallery should hold once the workers are done
GALLERY_STATE_KEY: Final[str] = "gallery:state"
GALLERY_STATE_MODEL: Final[str] = "model_version"
GALLERY_STATE_TOTAL: Final[str] = "total"
# Set of image ids the workers could not embed: skipped with a warning, not retried
GALLERY_FAILED_KEY: Final[str] = "gallery:failed"

### Qdrant

GALLERY_COLLECTION: Final[str] = "gallery"

### Object storage

QUERY_PREFIX: Final[str] = "queries"
GALLERY_PREFIX: Final[str] = "gallery"

### Ingestion: image validation

MAX_IMAGE_BYTES: Final[int] = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES: Final[frozenset[str]] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
})
MIN_SIDE_PX: Final[int] = 16
MAX_IMAGE_PIXELS: Final[int] = 50_000_000

### Search

DEFAULT_TOP_K: Final[int] = 10
MAX_TOP_K: Final[int] = 100

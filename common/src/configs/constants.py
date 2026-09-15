import os
from typing import Final

# Logging
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO").upper()

# Redis stream: backend publishes tasks, inference workers share one group
TASK_STREAM: Final[str] = "falcon:tasks"
INFERENCE_GROUP: Final[str] = "inference-workers"

# Single field inside a stream entry that carries the JSON payload
STREAM_PAYLOAD_FIELD: Final[str] = "payload"
# Approximate cap on stream length so Redis memory stays bounded
STREAM_MAXLEN: Final[int] = 100_000

# Result keys written by inference and polled by the backend
TASK_KEY: Final[str] = "task:{task_id}"
RESULT_KEY: Final[str] = "result:{task_id}"

# Long enough for a user to poll a slow query, short enough not to leak memory
TASK_TTL_SECONDS: Final[int] = 3600
RESULT_TTL_SECONDS: Final[int] = 3600

# Object storage prefixes
QUERY_PREFIX: Final[str] = "queries"
GALLERY_PREFIX: Final[str] = "gallery"

# Ingestion limits
MAX_IMAGE_BYTES: Final[int] = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES: Final[frozenset[str]] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
})
# A crop smaller than this carries no usable appearance signal
MIN_SIDE_PX: Final[int] = 16
# Pillow's decompression-bomb ceiling, raised to what a 4K camera can produce
MAX_IMAGE_PIXELS: Final[int] = 50_000_000

# Search defaults
DEFAULT_TOP_K: Final[int] = 10
MAX_TOP_K: Final[int] = 100

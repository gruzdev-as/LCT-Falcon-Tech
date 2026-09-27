import tempfile
from pathlib import Path
from typing import Final

# Sleep after an infrastructure failure before the next iteration
BACKOFF_SECONDS: Final[float] = 1.0

# Error text of the result written for a task that kept crashing its workers
POISON_ERROR: Final[str] = "task failed on every delivery attempt"

### Model artifacts: init downloads them into /weights, compose mounts it read-only

WEIGHTS_PATH: Final[Path] = Path("/weights/eva02.pt")

### Liveness

# Touched after every healthy iteration; the container healthcheck reads its mtime
HEARTBEAT_PATH: Final[Path] = Path(tempfile.gettempdir()) / "inference.alive"

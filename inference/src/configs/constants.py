from typing import Final

# Sleep after an infrastructure failure before the next iteration, so a Redis or
# Qdrant outage does not turn the loop into a busy spin
BACKOFF_SECONDS: Final[float] = 1.0

# Error text of the result written for a task that kept crashing its workers
POISON_ERROR: Final[str] = "task failed on every delivery attempt"

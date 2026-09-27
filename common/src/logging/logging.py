import logging
import sys

from common.src.configs.constants import LOG_LEVEL

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure root logging identically in every service."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)

    # httpx logs every request at INFO, and the Qdrant client speaks HTTP: one line per
    # search or count would drown the events that carry task_id.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # To ensure the same format of logging
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

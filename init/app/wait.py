import logging
import sys

import anyio

from common.src.exceptions import FalconError
from common.src.logging.logging import setup_logging
from common.src.qdrant.client import close_qdrant
from common.src.redis.client import close_redis
from init.src.configs.settings import get_settings
from init.src.wait import wait_for_gallery

setup_logging()
logger = logging.getLogger(__name__)


async def run() -> None:
    """Wait for the gallery once."""
    try:
        await wait_for_gallery(get_settings())
    finally:
        await close_redis()
        await close_qdrant()


def main() -> int:
    """Entry point of gallery-ready: 0 once the gallery is indexed, else 1."""
    logger.info("Waiting for the inference workers to index the gallery")
    try:
        anyio.run(run)
    except FalconError as exc:
        logger.error("Gallery not ready: %s %s", exc.message, exc.details)  # noqa: TRY400
        return 1
    except Exception:
        logger.exception("Gallery not ready")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

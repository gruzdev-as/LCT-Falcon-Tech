import argparse
import logging
import sys

import anyio

from common.src.exceptions import FalconError
from common.src.logging.logging import setup_logging
from init.src.configs.settings import get_settings
from init.src.pipeline import bootstrap, shutdown

setup_logging()
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bootstrap arguments."""
    parser = argparse.ArgumentParser(prog="init", description="One-shot bootstrap: artifacts, schema, gallery")
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload and rebuild instead of skipping what is already in place",
    )
    return parser.parse_args(argv)


async def run(force: bool) -> None:  # noqa: FBT001  # anyio.run passes positionally
    """Execute the bootstrap once."""
    settings = get_settings()
    if force:
        settings = settings.model_copy(update={"force": True})
    try:
        await bootstrap(settings)
    finally:
        await shutdown()


def main(argv: list[str] | None = None) -> int:
    """Entry point: returns the process exit code."""
    force = parse_args(argv).force
    logger.info("Bootstrap starting%s", " (forced)" if force else "")
    try:
        anyio.run(run, force)
    except FalconError as exc:
        logger.error("Bootstrap failed: %s %s", exc.message, exc.details)  # noqa: TRY400
        return 1
    except Exception:
        logger.exception("Bootstrap failed")
        return 1
    logger.info("Bootstrap done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

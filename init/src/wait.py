import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anyio

from common.src.configs.constants import (
    GALLERY_COLLECTION,
    GALLERY_FAILED_KEY,
    GALLERY_STATE_KEY,
    GALLERY_STATE_MODEL,
    GALLERY_STATE_TOTAL,
)
from common.src.exceptions import ValidationError
from common.src.qdrant import client as qdrant
from common.src.qdrant.gallery import MODEL_VERSION_FIELD
from common.src.redis.client import get_redis
from init.src.configs.settings import InitSettings

logger = logging.getLogger(__name__)

_FAILED_EXAMPLES = 10


@dataclass(frozen=True, slots=True)
class Progress:
    """How far the workers are through the gallery init queued."""

    model_version: str
    total: int
    indexed: int
    """Points in the collection carrying the expected weights."""

    failed: int
    """Images the workers skipped as broken."""

    @property
    def done(self) -> bool:
        """Every image is either indexed or given up on."""
        return self.indexed + self.failed >= self.total


async def read_progress() -> Progress:
    """Compare what init expects with what the collection holds.

    Raises:
        ValidationError: init has not recorded a gallery to wait for.
    """
    redis = get_redis()
    state = await redis.hgetall(GALLERY_STATE_KEY)
    if not state:
        msg = f"no {GALLERY_STATE_KEY} in Redis: init has not queued a gallery"
        raise ValidationError(msg)

    version = state[GALLERY_STATE_MODEL]
    return Progress(
        model_version=version,
        total=int(state[GALLERY_STATE_TOTAL]),
        indexed=await qdrant.count_points(GALLERY_COLLECTION, where=(MODEL_VERSION_FIELD, version)),
        failed=int(await redis.scard(GALLERY_FAILED_KEY)),
    )


async def wait_for_gallery(
    settings: InitSettings,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
) -> Progress:
    """Block until the workers have indexed the gallery, logging progress on the way.

    Broken images are reported and left out; nothing indexed at all is an error.

    Raises:
        ValidationError: init queued nothing, or no image could be indexed.
        TimeoutError: ``wait_timeout_s`` passed first.
    """
    started = clock()
    progress = await read_progress()
    first_indexed = progress.indexed
    last_indexed, last_change = progress.indexed, started

    while not progress.done:
        now = clock()
        _log_progress(progress, first_indexed, now - started)
        if progress.indexed != last_indexed:
            last_indexed, last_change = progress.indexed, now
        elif now - last_change >= settings.wait_stall_s:
            logger.warning(
                "No gallery progress for %.0fs: are the inference workers up, and do they serve weights %s?",
                now - last_change,
                progress.model_version[:12],
            )
            last_change = now
        if settings.wait_timeout_s and now - started >= settings.wait_timeout_s:
            msg = f"gallery not indexed after {settings.wait_timeout_s:.0f}s: {progress.indexed}/{progress.total}"
            raise TimeoutError(msg)
        await sleep(settings.wait_poll_s)
        progress = await read_progress()

    await _report(progress)
    return progress


def _log_progress(progress: Progress, first_indexed: int, elapsed: float) -> None:
    rate = (progress.indexed - first_indexed) / elapsed if elapsed > 0 else 0.0
    remaining = progress.total - progress.indexed - progress.failed
    eta = f"{remaining / rate:.0f}s" if rate > 0 else "unknown"
    logger.info(
        "Gallery indexing: %d/%d (%.0f%%), %d skipped, %.1f img/s, ETA %s",
        progress.indexed,
        progress.total,
        100 * progress.indexed / progress.total if progress.total else 100.0,
        progress.failed,
        rate,
        eta,
    )


async def _report(progress: Progress) -> None:
    if progress.total and not progress.indexed:
        msg = "the workers indexed no gallery image at all; check their logs and weights"
        raise ValidationError(msg, details={"total": progress.total, "failed": progress.failed})
    if progress.failed:
        examples = await get_redis().srandmember(GALLERY_FAILED_KEY, _FAILED_EXAMPLES)
        logger.warning(
            "Gallery ready without %d broken images (e.g. %s); see the inference logs for why",
            progress.failed,
            ", ".join(sorted(examples)),
        )
    logger.info("Gallery ready: %d images indexed with weights %s", progress.indexed, progress.model_version[:12])

import logging
import signal

import anyio

from common.src.logging.logging import setup_logging
from common.src.qdrant.client import close_qdrant
from common.src.redis.client import close_redis
from inference.src.configs.settings import get_settings
from inference.src.models.factory import build_embedder, build_refusal
from inference.src.processor import Processor
from inference.src.worker import InferenceWorker

setup_logging()
logger = logging.getLogger(__name__)


async def _stop_on_signal(worker: InferenceWorker) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        async for signum in signals:
            logger.info("Received %s, finishing the task in flight", signal.Signals(signum).name)
            worker.stop()
            return


async def main() -> None:
    settings = get_settings()
    embedder = build_embedder(settings)
    worker = InferenceWorker(Processor(embedder, build_refusal(settings)), settings)

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(_stop_on_signal, worker)
            await worker.run()
            group.cancel_scope.cancel()
    finally:
        await close_redis()
        await close_qdrant()


if __name__ == "__main__":
    anyio.run(main)

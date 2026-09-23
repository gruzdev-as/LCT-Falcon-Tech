import io
import uuid
from collections.abc import Awaitable, Callable

import numpy as np
from PIL import Image

from common.src.configs.schemas import BBox, EmbeddingTask

DIM = 16
BOX = BBox(x=10, y=10, width=64, height=48)

type GalleryAdd = Callable[..., Awaitable[str]]
type Submit = Callable[..., Awaitable[EmbeddingTask]]


def make_image(seed: int, size: tuple[int, int] = (128, 96), image_format: str = "PNG") -> bytes:
    """Encode a noise image; different seeds give visually unrelated images."""
    pixels = np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=image_format)
    return buffer.getvalue()


def make_task(image_path: str = "queries/q.png", bbox: BBox = BOX, top_k: int = 5) -> EmbeddingTask:
    return EmbeddingTask(task_id=uuid.uuid4().hex, image_path=image_path, bbox=bbox, top_k=top_k)

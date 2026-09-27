import hashlib
import io
import uuid
from collections.abc import Awaitable, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

from common.src.configs.schemas import BBox, EmbeddingTask, GalleryTask
from common.src.exceptions import ValidationError

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


def make_gallery_task(image_id: str = "g-1", model_version: str = "fake", bbox: BBox = BOX) -> GalleryTask:
    return GalleryTask(
        task_id=uuid.uuid4().hex,
        image_id=image_id,
        image_path=f"gallery/{image_id}.png",
        bbox=bbox,
        vehicle_id="car-1",
        camera_id="cam-1",
        model_version=model_version,
    )


class FakeEmbedder:
    """Stands in for the ReID model: the pixels of the crop seed a random vector.

    Identical crops embed identically and different ones nearly orthogonally, which is
    all the pipeline needs to be tested without torch weights.
    """

    name = "fake"
    version = "fake"

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, data: bytes, bbox: BBox) -> np.ndarray:
        """Crop by the box and hash the pixels, failing the way the real model does."""
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                crop = image.crop(bbox.clamp(*image.size).to_xyxy())
        except (UnidentifiedImageError, OSError) as exc:
            msg = "stored image cannot be decoded"
            raise ValidationError(msg) from exc
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        seed = int.from_bytes(hashlib.sha256(crop.tobytes()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

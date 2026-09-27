import numpy as np

from common.src.configs.schemas import BBox
from inference.src.models.base import Embedder
from inference.src.pipeline.postprocess import l2_normalize


def embed_vector(embedder: Embedder, data: bytes, bbox: BBox) -> np.ndarray:
    """Embed one vehicle and L2-normalize it; the one path both searches and the gallery take.

    Blocking; callers run it in a worker thread.

    Raises:
        ValidationError: the bytes do not decode, or the box degenerates on the image.
        RuntimeError: the embedder returned a vector of the wrong shape.
    """
    vector = l2_normalize(embedder.embed(data, bbox))
    if vector.shape != (embedder.dim,):
        msg = f"embedder returned shape {vector.shape}, expected ({embedder.dim},)"
        raise RuntimeError(msg)
    return vector

from typing import Protocol

import numpy as np

from common.src.configs.schemas import BBox


class Embedder(Protocol):
    """An appearance model: one stored frame and its vehicle box in, one vector out.

    The embedder owns decoding, cropping and preprocessing, because they are part of
    the model's recipe: the context around the box, the input size and the
    normalization have to match what the gallery was embedded with.
    """

    name: str
    """Identifier reported in ``SearchResult.model_name``."""

    dim: int
    """Length of the vector ``embed`` returns; the Qdrant collection is sized by it."""

    def embed(self, data: bytes, bbox: BBox) -> np.ndarray:
        """Embed the vehicle inside one encoded image.

        Blocking; the worker calls it off the event loop.

        Args:
            data: encoded image bytes as stored at ingestion.
            bbox: vehicle box in absolute pixels of the decoded frame.

        Returns:
            float32 array of shape ``(dim,)``.

        Raises:
            ValidationError: the bytes do not decode, or the box degenerates on the image.
        """
        ...

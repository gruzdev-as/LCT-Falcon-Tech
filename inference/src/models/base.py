from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    """An appearance model: one vehicle crop in, one vector out."""

    name: str
    """Identifier reported in ``SearchResult.model_name``."""

    dim: int
    """Length of the vector ``embed`` returns; the Qdrant collection is sized by it."""

    input_size: tuple[int, int]
    """(width, height) the crop is resized to before ``embed``."""

    def embed(self, image: Image.Image) -> np.ndarray:
        """Embed one preprocessed RGB crop.

        Blocking; the worker calls it off the event loop.

        Returns:
            float32 array of shape ``(dim,)``. Need not be normalized — the pipeline
            normalizes every vector itself.
        """
        ...

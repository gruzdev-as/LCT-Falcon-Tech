import hashlib

import numpy as np
from PIL import Image

_DEFAULT_INPUT_SIZE = (224, 224)


class StubEmbedder:
    """Deterministic stand-in until a trained ReID model exists."""

    name = "stub"

    def __init__(self, dim: int, input_size: tuple[int, int] = _DEFAULT_INPUT_SIZE) -> None:
        self.dim = dim
        self.input_size = input_size

    def embed(self, image: Image.Image) -> np.ndarray:
        """Return a pixel-seeded random vector."""
        seed = int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

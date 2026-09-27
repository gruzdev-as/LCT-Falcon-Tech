import hashlib

import numpy as np

from common.src.configs.schemas import BBox
from inference.src.pipeline.crop import crop_vehicle
from inference.src.pipeline.preprocess import preprocess

_DEFAULT_INPUT_SIZE = (224, 224)


class StubEmbedder:
    """Deterministic stand-in for tests and a torch-free local stack."""

    name = "stub"

    def __init__(self, dim: int, input_size: tuple[int, int] = _DEFAULT_INPUT_SIZE) -> None:
        self.dim = dim
        self.input_size = input_size

    def embed(self, data: bytes, bbox: BBox) -> np.ndarray:
        """Return a vector seeded by the pixels of the preprocessed crop.

        Raises:
            ValidationError: the bytes do not decode, or the box degenerates on the image.
        """
        crop = preprocess(crop_vehicle(data, bbox), self.input_size)
        seed = int.from_bytes(hashlib.sha256(crop.tobytes()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

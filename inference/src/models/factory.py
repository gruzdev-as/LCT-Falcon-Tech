from collections.abc import Callable

from inference.src.configs.settings import InferenceSettings
from inference.src.models.base import Embedder
from inference.src.models.stub import StubEmbedder

_EMBEDDERS: dict[str, Callable[[InferenceSettings], Embedder]] = {
    "stub": lambda settings: StubEmbedder(dim=settings.embedding_dim),
}


def build_embedder(settings: InferenceSettings) -> Embedder:
    """Instantiate — and so load — the configured embedding model.

    Raises:
        ValueError: no backend is registered under ``settings.embedder``.
    """
    try:
        factory = _EMBEDDERS[settings.embedder]
    except KeyError:
        msg = f"unknown embedder {settings.embedder!r}, expected one of {sorted(_EMBEDDERS)}"
        raise ValueError(msg) from None
    return factory(settings)

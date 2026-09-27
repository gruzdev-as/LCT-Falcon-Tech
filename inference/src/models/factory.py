import logging
from collections.abc import Callable

import torch

from inference.src.configs.refusal import load_refusal_preset
from inference.src.configs.settings import InferenceSettings
from inference.src.models.base import Embedder
from inference.src.models.refusal import CosineRefusal, Refusal
from inference.src.models.refusal_head import HeadRefusal
from inference.src.models.reid import ReIDEmbedder
from inference.src.models.stub import StubEmbedder

logger = logging.getLogger(__name__)


def _reid(settings: InferenceSettings) -> Embedder:
    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    return ReIDEmbedder(settings.weights_path, device=settings.device, compile_model=settings.compile)


_EMBEDDERS: dict[str, Callable[[InferenceSettings], Embedder]] = {
    "stub": lambda settings: StubEmbedder(dim=settings.embedding_dim),
    "eva02": _reid,
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


def build_refusal(settings: InferenceSettings) -> Refusal:
    """Build the rejection rule: training's served preset, unless a plain threshold overrides it.

    Raises:
        ValueError: the preset is invalid, or the CatBoost head it needs is missing.
    """
    if settings.reject_threshold is not None:
        logger.info("Refusal: cosine >= %.4f from INFERENCE_REJECT_THRESHOLD", settings.reject_threshold)
        return CosineRefusal(settings.reject_threshold)

    preset = load_refusal_preset(settings.refusal_config)
    if preset.kind == "threshold":
        assert preset.cosine_threshold is not None  # noqa: S101  # validated by the loader
        logger.info("Refusal: cosine >= %.4f from %s", preset.cosine_threshold, settings.refusal_config)
        return CosineRefusal(preset.cosine_threshold)

    logger.info("Refusal: %s from %s", preset.kind, settings.refusal_config)
    return HeadRefusal(preset, settings.refusal_model_path)

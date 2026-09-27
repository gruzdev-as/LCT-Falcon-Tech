import logging

import torch

from inference.src.configs.refusal import load_refusal_preset
from inference.src.configs.settings import InferenceSettings
from inference.src.models.base import Embedder
from inference.src.models.refusal import CosineRefusal, Refusal
from inference.src.models.refusal_head import HeadRefusal
from inference.src.models.reid import ReIDEmbedder

logger = logging.getLogger(__name__)


def build_embedder(settings: InferenceSettings) -> Embedder:
    """Load the ReID model once, for every task the worker will take."""
    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    return ReIDEmbedder(settings.weights_path, device=settings.device, compile_model=settings.compile)


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

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from augmentations import build_transforms
from dataset.images import crop_record, decode_rgb, preprocess_record
from models import ReIDModel
from models.kernels import prepare_inference_model
from modules.inference import embed_tensor, fuse_context_views, tta_context_pcts
from omegaconf import DictConfig, OmegaConf
from PIL import Image, UnidentifiedImageError

from common.src.configs.constants import MAX_IMAGE_PIXELS
from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Same decompression-bomb ceiling the backend validated against
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

SERVING_FORMAT = "reid-serving"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class ReIDEmbedder:
    """The trained ReID model, run through the training submodule's own code."""

    def __init__(self, weights_path: Path, *, device: str = "auto", compile_model: bool = False) -> None:
        """Load the checkpoint once; every task then reuses the model.

        Args:
            weights_path: the ``reid-serving`` ``.pt`` file.
            device: ``auto`` picks cuda when available, else cpu.
            compile_model: ``torch.compile`` the model.

        Raises:
            FileNotFoundError: no file at ``weights_path``.
            ValueError: the file is not a serving checkpoint, or its weights do not fit the architecture.
        """
        started = time.monotonic()
        self.device = torch.device(_pick_device(device))
        # Deterministic cuBLAS needs this before the first cuda call; training sets the same.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        blob = _load_checkpoint(weights_path)
        self.cfg = _serving_config(blob, self.device, compile_model=compile_model)

        model = ReIDModel(self.cfg, initialize_pretrained=False)
        _load_state(model, blob["state_dict"])
        model.to(self.device).eval()
        # Also applies the runtime flags (determinism, TF32 off) the gallery was embedded under.
        self._model = prepare_inference_model(model, self.cfg)

        self._transform = build_transforms(self.cfg)
        self._contexts = tta_context_pcts(self.cfg.eval.tta, self.cfg.data.context_pct)

        self.name = str(self.cfg.get("name") or weights_path.stem)
        self.dim = int(model.embedding_dim)
        self.version = _sha256(weights_path)
        logger.info(
            "Loaded %s from %s: device=%s precision=%s dim=%d contexts=%s sha256=%s in %.1fs",
            self.name,
            weights_path,
            self.device,
            self.cfg.eval.precision,
            self.dim,
            self._contexts,
            self.version[:12],
            time.monotonic() - started,
        )

    def embed(self, data: bytes, bbox: BBox) -> np.ndarray:
        """Embed one vehicle exactly as ``training.modules.inference.embed_frame`` does.

        Raises:
            ValidationError: the bytes do not decode, or the box does not cut anything out of the image.
        """
        try:
            image = decode_rgb(data, backend="pil")
        except (UnidentifiedImageError, OSError) as exc:
            msg = "stored image cannot be decoded"
            raise ValidationError(msg) from exc

        box = (bbox.x, bbox.y, bbox.width, bbox.height)
        views = [self._embed_view(image, box, context) for context in self._contexts]
        return fuse_context_views(views)[0].astype(np.float32)

    def _embed_view(self, image: Image.Image, box: tuple[float, ...], context: float) -> np.ndarray:
        """One context crop through the model, with the checkpoint's TTA."""
        try:
            crop = crop_record(image, box, context)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        batch = preprocess_record(crop, self._transform).unsqueeze(0).to(self.device)
        vector = embed_tensor(
            self._model,
            batch,
            self.device,
            precision=str(self.cfg.eval.precision),
            tta=self.cfg.eval.tta,
            model_cfg=self.cfg.model,
        )
        return vector.cpu().numpy()


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)  # noqa: S614
    if not isinstance(blob, dict) or blob.get("format") != SERVING_FORMAT:
        msg = f"{path} is not a {SERVING_FORMAT} checkpoint; export it with training/scripts/export_serving.py"
        raise ValueError(msg)
    if not blob.get("state_dict"):
        msg = f"{path} has no state_dict"
        raise ValueError(msg)
    return blob


def _serving_config(blob: dict[str, Any], device: torch.device, *, compile_model: bool) -> DictConfig:
    """The training config saved in the checkpoint, with only the runtime knobs overridden."""
    cfg = OmegaConf.create(blob["cfg"])
    assert isinstance(cfg, DictConfig)  # noqa: S101  # the payload stores a mapping
    on_cuda = device.type == "cuda"
    OmegaConf.update(cfg, "eval.device", str(device), force_add=True)
    OmegaConf.update(cfg, "eval.precision", "bf16" if on_cuda else "fp32", force_add=True)
    OmegaConf.update(cfg, "eval.fast_kernels", False, force_add=True)
    OmegaConf.update(cfg, "data.decode_backend", "pil", force_add=True)
    OmegaConf.update(cfg, "model.compile", bool(compile_model and on_cuda), force_add=True)
    return cfg


def _load_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load the weights, tolerating only the token the SSL pretraining adds."""
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.endswith("mask_token")]
    if missing or result.unexpected_keys:
        msg = f"checkpoint does not fit the model: missing={missing}, unexpected={result.unexpected_keys}"
        raise ValueError(msg)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()

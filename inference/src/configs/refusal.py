import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

type RefusalKind = Literal["threshold", "model", "ensemble"]
_KINDS: frozenset[str] = frozenset({"threshold", "model", "ensemble"})


@dataclass(frozen=True, slots=True)
class RefusalPreset:
    """How a search decides that the gallery holds no match, as training tuned it."""

    kind: RefusalKind
    """``threshold``: cosine only; ``model``: CatBoost only; ``ensemble``: both must accept."""

    cosine_threshold: float | None
    model_threshold: float | None
    k: int
    """How many nearest gallery vectors the CatBoost features are built from."""

    with_embeddings: bool
    """Whether the features include the query and the top-1 gallery vector."""


def load_refusal_preset(path: Path) -> RefusalPreset:
    """Read and validate a training refusal preset.

    Raises:
        ValueError: the file is missing, names an unknown kind, or lacks a threshold its kind needs.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        msg = f"refusal preset {path} does not exist; set INFERENCE_REFUSAL_CONFIG or INFERENCE_REJECT_THRESHOLD"
        raise ValueError(msg) from None
    if not isinstance(raw, dict) or raw.get("kind") not in _KINDS:
        msg = f"{path} is not a refusal preset this worker serves (kind must be one of {sorted(_KINDS)})"
        raise ValueError(msg)

    kind = raw["kind"]
    cosine = _probability(raw, "cosine_threshold", path, required=kind in {"threshold", "ensemble"})
    model = _probability(raw, "model_threshold", path, required=kind in {"model", "ensemble"})

    k = raw.get("k", 10)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        msg = f"{path} has k={k!r}, expected a positive integer"
        raise ValueError(msg)

    return RefusalPreset(
        kind=kind,
        cosine_threshold=cosine,
        model_threshold=model,
        k=k,
        with_embeddings=bool(raw.get("with_embeddings", True)),
    )


def _probability(raw: dict, key: str, path: Path, *, required: bool) -> float | None:
    value = raw.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool) or not 0 <= value <= 1:
        msg = f"{path} needs {key} in [0, 1] for kind {raw['kind']!r}, got {value!r}"
        raise ValueError(msg)
    return float(value)

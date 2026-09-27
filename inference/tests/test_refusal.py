from pathlib import Path

import pytest

from inference.src.configs.refusal import load_refusal_preset
from inference.src.configs.settings import InferenceSettings
from inference.src.models.factory import build_refusal
from inference.src.models.refusal import CosineRefusal

PRESETS = Path("training/configs/refusal")


def _preset(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "preset.yaml"
    path.write_text(body)
    return path


def test_reads_the_preset_training_serves() -> None:
    preset = load_refusal_preset(PRESETS / "eva02_ensemble.yaml")

    assert preset.kind == "ensemble"
    assert 0 < preset.cosine_threshold < 1
    assert 0 < preset.model_threshold < 1
    assert preset.k >= 1


def test_the_default_preset_is_the_one_training_serves() -> None:
    assert InferenceSettings().refusal_config == PRESETS / "eva02_ensemble.yaml"


def test_reads_every_training_preset() -> None:
    kinds = {load_refusal_preset(path).kind for path in PRESETS.glob("eva02_*.yaml")}
    assert kinds == {"threshold", "model", "ensemble"}


def test_a_threshold_preset_needs_no_catboost(tmp_path: Path) -> None:
    preset = _preset(tmp_path, "kind: threshold\ncosine_threshold: 0.61\n")

    refusal = build_refusal(InferenceSettings(refusal_config=preset))

    assert isinstance(refusal, CosineRefusal)
    assert refusal.threshold == pytest.approx(0.61)


def test_an_explicit_threshold_overrides_the_preset(tmp_path: Path) -> None:
    refusal = build_refusal(InferenceSettings(reject_threshold=0.3, refusal_config=tmp_path / "missing.yaml"))

    assert isinstance(refusal, CosineRefusal)
    assert refusal.threshold == 0.3


def test_an_empty_variable_means_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_REJECT_THRESHOLD", "")
    assert InferenceSettings().reject_threshold is None


def test_a_missing_preset_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_refusal_preset(tmp_path / "missing.yaml")


def test_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        load_refusal_preset(_preset(tmp_path, "kind: none\n"))


def test_an_ensemble_needs_both_thresholds(tmp_path: Path) -> None:
    preset = _preset(tmp_path, "kind: ensemble\ncosine_threshold: 0.5\nmodel_threshold: null\n")
    with pytest.raises(ValueError, match="model_threshold"):
        load_refusal_preset(preset)


def test_a_threshold_outside_the_unit_range_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cosine_threshold"):
        load_refusal_preset(_preset(tmp_path, "kind: threshold\ncosine_threshold: 1.5\n"))

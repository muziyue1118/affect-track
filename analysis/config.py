from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    resample_hz: int = 200
    notch_hz: float = 50.0
    bandpass_hz: tuple[float, float] = (1.0, 45.0)
    reference: str = "average"
    montage: str = "standard_1020"
    reject_amplitude_uv: float = 150.0
    run_ica: bool = False


@dataclass(frozen=True)
class SegmentationConfig:
    trim_start_s: float = 30.0
    trim_end_s: float = 10.0
    duration_s: float | str = "full"
    min_trial_s: float = 60.0
    window_s: float = 4.0
    window_overlap_s: float = 0.0


@dataclass(frozen=True)
class FeatureConfig:
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 14.0),
            "beta": (14.0, 31.0),
            "gamma": (31.0, 45.0),
        }
    )


@dataclass(frozen=True)
class EvaluationConfig:
    primary_task: str = "category"
    secondary_tasks: tuple[str, ...] = ("valence_binary", "arousal_binary")
    split_mode: str = "loso"
    tune_hyperparameters: bool = False
    class_labels: tuple[str, ...] = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class ModelConfig:
    feature_models: tuple[str, ...] = (
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
        "random_forest",
    )
    deep_models: tuple[str, ...] = ("shallow_convnet",)
    deep_epochs: int = 80
    deep_batch_size: int = 32
    deep_learning_rate: float = 0.001
    deep_patience: int = 10
    deep_device: str = "auto"


@dataclass(frozen=True)
class EEGConfig:
    data_dir: Path = Path("data/eeg_data")
    labels_csv: Path = Path("data/offline_records.csv")
    output_dir: Path = Path("outputs/eeg_runs")
    random_seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: str | Path) -> EEGConfig:
    config_path = Path(path)
    raw = _load_yaml_like(config_path)
    base_dir = config_path.parent.parent if config_path.parent.name == "analysis" else Path.cwd()

    def path_value(key: str, default: str) -> Path:
        value = Path(str(raw.get(key, default)))
        return value if value.is_absolute() else base_dir / value

    preprocessing = raw.get("preprocessing", {})
    segmentation = raw.get("segmentation", {})
    features = raw.get("features", {})
    evaluation = raw.get("evaluation", {})
    models = raw.get("models", {})
    default_models = ModelConfig()
    bands = _parse_bands(features.get("bands", {}))

    return EEGConfig(
        data_dir=path_value("data_dir", "data/eeg_data"),
        labels_csv=path_value("labels_csv", "data/offline_records.csv"),
        output_dir=path_value("output_dir", "outputs/eeg_runs"),
        random_seed=int(raw.get("random_seed", 42)),
        preprocessing=PreprocessingConfig(
            resample_hz=int(preprocessing.get("resample_hz", 200)),
            notch_hz=float(preprocessing.get("notch_hz", 50)),
            bandpass_hz=tuple(float(v) for v in preprocessing.get("bandpass_hz", [1, 45])),
            reference=str(preprocessing.get("reference", "average")),
            montage=str(preprocessing.get("montage", "standard_1020")),
            reject_amplitude_uv=float(preprocessing.get("reject_amplitude_uv", 150)),
            run_ica=bool(preprocessing.get("run_ica", False)),
        ),
        segmentation=SegmentationConfig(
            trim_start_s=float(segmentation.get("trim_start_s", 30)),
            trim_end_s=float(segmentation.get("trim_end_s", 10)),
            duration_s=_duration_value(segmentation.get("duration_s", "full")),
            min_trial_s=float(segmentation.get("min_trial_s", 60)),
            window_s=float(segmentation.get("window_s", 4)),
            window_overlap_s=float(segmentation.get("window_overlap_s", 0)),
        ),
        features=FeatureConfig(bands=bands or FeatureConfig().bands),
        evaluation=EvaluationConfig(
            primary_task=str(evaluation.get("primary_task", "category")),
            secondary_tasks=tuple(evaluation.get("secondary_tasks", ["valence_binary", "arousal_binary"])),
            split_mode=str(evaluation.get("split_mode", "loso")),
            tune_hyperparameters=bool(evaluation.get("tune_hyperparameters", False)),
            class_labels=tuple(evaluation.get("class_labels", ["negative", "neutral", "positive"])),
        ),
        models=ModelConfig(
            feature_models=tuple(models.get("feature_models", list(default_models.feature_models))),
            deep_models=tuple(models.get("deep_models", list(default_models.deep_models))),
            deep_epochs=int(models.get("deep_epochs", 80)),
            deep_batch_size=int(models.get("deep_batch_size", 32)),
            deep_learning_rate=float(models.get("deep_learning_rate", 0.001)),
            deep_patience=int(models.get("deep_patience", 10)),
            deep_device=str(models.get("deep_device", "auto")),
        ),
    )


def _parse_bands(raw_bands: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {name: (float(bounds[0]), float(bounds[1])) for name, bounds in raw_bands.items()}


def _duration_value(value: Any) -> float | str:
    if isinstance(value, str) and value.lower() == "full":
        return "full"
    return float(value)


def _load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, _, value = line.strip().partition(":")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            current = stack[-1][1]
            if value.strip() == "":
                child: dict[str, Any] = {}
                current[key] = child
                stack.append((indent, child))
            else:
                current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        content = value[1:-1].strip()
        if not content:
            return []
        return [_parse_scalar(part.strip()) for part in content.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


InputKind = Literal["raw", "de", "de_sequence"]
Protocol = Literal["supervised", "source_dg", "transductive_da"]


CLASSICAL_MODEL_NAMES = ("logistic_regression", "linear_svm", "rbf_svm", "random_forest")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_kind: InputKind
    protocol: Protocol = "supervised"
    supports_channels: tuple[int, ...] | None = None
    requires_sequence: bool = False
    requires_target_unlabeled: bool = False
    default_params: dict[str, object] = field(default_factory=dict)
    source: str = "local"


TORCH_MODEL_SPECS: dict[str, ModelSpec] = {
    "shallow_convnet": ModelSpec("shallow_convnet", "raw", source="affect-track"),
    "EEGNet": ModelSpec("EEGNet", "raw", source="LibEER"),
    "TSception": ModelSpec("TSception", "raw", source="LibEER"),
    "ACRNN": ModelSpec("ACRNN", "raw", source="LibEER"),
    "FBSTCNet": ModelSpec("FBSTCNet", "raw", source="LibEER"),
    "DGCNN": ModelSpec("DGCNN", "de", supports_channels=(32, 62), source="LibEER"),
    "GCBNet": ModelSpec("GCBNet", "de", supports_channels=(32, 62), source="LibEER"),
    "GCBNet_BLS": ModelSpec("GCBNet_BLS", "de", supports_channels=(32, 62), source="LibEER"),
    "CDCN": ModelSpec("CDCN", "de", source="LibEER"),
    "DBN": ModelSpec("DBN", "de", source="LibEER", default_params={"pretrain_epochs": 1}),
    "HSLT": ModelSpec("HSLT", "de", supports_channels=(32, 62), source="LibEER"),
    "RGNN": ModelSpec("RGNN", "de", supports_channels=(32, 62), source="LibEER"),
    "RGNN_official": ModelSpec("RGNN_official", "de", supports_channels=(32, 62), source="LibEER"),
    "STRNN": ModelSpec("STRNN", "de_sequence", requires_sequence=True, source="LibEER"),
    "CoralDgcnn": ModelSpec("CoralDgcnn", "de", "source_dg", supports_channels=(32, 62), source="LibEER"),
    "DannDgcnn": ModelSpec("DannDgcnn", "de", "source_dg", supports_channels=(32, 62), source="LibEER"),
    "BiDANN": ModelSpec(
        "BiDANN",
        "de_sequence",
        "transductive_da",
        supports_channels=(32, 62),
        requires_sequence=True,
        requires_target_unlabeled=True,
        source="LibEER",
    ),
    "R2GSTNN": ModelSpec(
        "R2GSTNN",
        "de_sequence",
        "transductive_da",
        supports_channels=(32, 62),
        requires_sequence=True,
        requires_target_unlabeled=True,
        source="LibEER",
    ),
    "MsMDA": ModelSpec("MsMDA", "de", "transductive_da", requires_target_unlabeled=True, source="LibEER"),
    "NSAL_DGAT": ModelSpec(
        "NSAL_DGAT",
        "de",
        "transductive_da",
        supports_channels=(32, 62),
        requires_target_unlabeled=True,
        source="LibEER",
    ),
    "PRRL": ModelSpec("PRRL", "de", "transductive_da", requires_target_unlabeled=True, source="LibEER"),
}

DEEP_MODEL_NAMES = tuple(TORCH_MODEL_SPECS)


def list_models(protocol: str | None = None, input_kind: str | None = None) -> list[ModelSpec]:
    specs = list(TORCH_MODEL_SPECS.values())
    if protocol:
        specs = [spec for spec in specs if spec.protocol == protocol]
    if input_kind and input_kind != "auto":
        specs = [spec for spec in specs if spec.input_kind == input_kind]
    return specs


def get_model_spec(name: str) -> ModelSpec:
    if name not in TORCH_MODEL_SPECS:
        raise ValueError(f"Unknown torch model: {name}. Available: {', '.join(DEEP_MODEL_NAMES)}")
    return TORCH_MODEL_SPECS[name]


def build_classical_model(name: str, random_seed: int):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for feature classification.") from exc

    builders = {
        "logistic_regression": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed)),
            ]
        ),
        "linear_svm": lambda: Pipeline(
            [("scale", StandardScaler()), ("clf", SVC(kernel="linear", class_weight="balanced"))]
        ),
        "rbf_svm": lambda: Pipeline(
            [("scale", StandardScaler()), ("clf", SVC(kernel="rbf", class_weight="balanced", gamma="scale"))]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=1,
        ),
    }
    if name not in builders:
        raise ValueError(f"Unknown classifier: {name}. Available: {', '.join(CLASSICAL_MODEL_NAMES)}")
    return builders[name]()


def build_deep_model(name: str, n_channels: int, n_classes: int, n_samples: int):
    return build_torch_model(name, n_channels=n_channels, n_classes=n_classes, n_samples=n_samples)


def build_torch_model(
    name: str,
    *,
    n_channels: int,
    n_classes: int,
    n_samples: int | None = None,
    n_bands: int = 5,
    sequence_length: int = 9,
    num_domains: int = 2,
):
    spec = get_model_spec(name)
    if spec.supports_channels is not None and n_channels not in spec.supports_channels:
        raise ValueError(
            f"{name} supports channels {spec.supports_channels}, got {n_channels}. "
            "This model is skipped for the current EEG montage."
        )
    if spec.input_kind == "raw":
        if n_samples is None:
            raise ValueError(f"{name} requires n_samples for raw EEG input.")
        from analysis.libeer_models.raw_models import build_raw_model

        return build_raw_model(name, n_channels=n_channels, n_classes=n_classes, n_samples=n_samples)
    if spec.input_kind == "de_sequence":
        from analysis.libeer_models.sequence_models import build_sequence_model

        return build_sequence_model(
            name,
            n_channels=n_channels,
            n_bands=n_bands,
            n_classes=n_classes,
            sequence_length=sequence_length,
            num_domains=num_domains,
        )
    if spec.protocol == "transductive_da":
        from analysis.libeer_models.domain_models import build_domain_adaptation_model

        return build_domain_adaptation_model(
            name,
            input_kind=spec.input_kind,
            n_channels=n_channels,
            n_bands=n_bands,
            n_classes=n_classes,
            num_domains=num_domains,
        )
    if spec.protocol == "source_dg":
        from analysis.libeer_models.domain_models import build_source_generalization_model

        return build_source_generalization_model(
            name,
            n_channels=n_channels,
            n_bands=n_bands,
            n_classes=n_classes,
            num_domains=num_domains,
        )
    from analysis.libeer_models.de_models import build_de_model

    return build_de_model(name, n_channels=n_channels, n_bands=n_bands, n_classes=n_classes)

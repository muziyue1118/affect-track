from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

from analysis.Net import CLASSICAL_MODEL_NAMES, build_classical_model
from analysis.config import EEGConfig
from analysis.splits import EvaluationSplit, assert_no_split_leakage


def run_feature_classification(
    X,
    y: Sequence[str],
    subjects: Sequence[str],
    trial_ids: Sequence[str],
    splits: Sequence[EvaluationSplit],
    config: EEGConfig,
    output_dir: Path,
    feature_name: str,
    classifier: str | None = None,
) -> list[dict[str, object]]:
    try:
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn and NumPy are required for feature classification.") from exc

    labels = sorted(set(y))
    results: list[dict[str, object]] = []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    model_names = _resolve_model_names(classifier, config.models.feature_models)
    for split in splits:
        assert_no_split_leakage(split, subjects, trial_ids, config.evaluation.split_mode)
        y_train = y_arr[list(split.train_indices)]
        y_test = y_arr[list(split.test_indices)]
        if len(set(y_train)) < 2 or len(set(y_test)) < 1:
            continue
        for model_name in model_names:
            model = build_classical_model(model_name, random_seed=config.random_seed)
            model.fit(X_arr[list(split.train_indices)], y_train)
            pred = model.predict(X_arr[list(split.test_indices)])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                warnings.filterwarnings("ignore", message="A single label was found")
                balanced_accuracy = balanced_accuracy_score(y_test, pred)
                macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
                matrix = confusion_matrix(y_test, pred, labels=labels).tolist()
            results.append(
                {
                    "feature": feature_name,
                    "model": model_name,
                    "split": split.name,
                    "balanced_accuracy": float(balanced_accuracy),
                    "macro_f1": float(macro_f1),
                    "n_train": int(len(split.train_indices)),
                    "n_test": int(len(split.test_indices)),
                    "test_classes": sorted(set(y_test)),
                    "labels": labels,
                    "confusion_matrix": matrix,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{feature_name}_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def _resolve_model_names(classifier: str | None, configured: Sequence[str]) -> list[str]:
    if classifier and classifier != "all":
        if classifier not in CLASSICAL_MODEL_NAMES:
            raise ValueError(f"Unknown classifier: {classifier}. Available: {', '.join(CLASSICAL_MODEL_NAMES)}")
        return [classifier]
    return [name for name in configured if name in CLASSICAL_MODEL_NAMES]

import numpy as np
import pytest
import shutil
from pathlib import Path
from types import SimpleNamespace

from analysis.online_training import _binary_metrics, _guard_output_dir, _select_constrained_category_holdout


def test_binary_metrics_include_train_accuracy_fields() -> None:
    metrics = _binary_metrics(
        np.asarray([0, 0, 1, 1], dtype="int64"),
        np.asarray([0.1, 0.7, 0.8, 0.2], dtype="float64"),
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["prediction_counts"] == {"low": 2, "high": 2}


def test_constrained_holdout_keeps_binary_coverage_for_both_tasks() -> None:
    trials = [
        SimpleNamespace(trial_id="pos_a", video_name="positive_1.mp4", category="positive", valence=5, arousal=5),
        SimpleNamespace(trial_id="pos_b", video_name="positive_2.mp4", category="positive", valence=4, arousal=4),
        SimpleNamespace(trial_id="neg_a", video_name="negative_1.mp4", category="negative", valence=1, arousal=2),
        SimpleNamespace(trial_id="neg_b", video_name="negative_2.mp4", category="negative", valence=2, arousal=5),
        SimpleNamespace(trial_id="neg_c", video_name="negative_3.mp4", category="negative", valence=1, arousal=2),
    ]

    holdout = _select_constrained_category_holdout(
        trials,
        positive_count=1,
        negative_count=1,
        split_seed=42,
    )

    assert {trial.category for trial in holdout} == {"positive", "negative"}
    assert {trial.trial_id for trial in holdout} == {"pos_a", "neg_a"} or {
        trial.trial_id for trial in holdout
    } == {"pos_b", "neg_a"}


def test_output_dir_refuses_to_overwrite_existing_model() -> None:
    tmp_path = Path("outputs/test_tmp_online_training_guard")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    (tmp_path / "valence_tsception.pt").write_bytes(b"model")

    try:
        with pytest.raises(FileExistsError, match="already contains online model artifacts"):
            _guard_output_dir(tmp_path, overwrite=False)

        _guard_output_dir(tmp_path, overwrite=True)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

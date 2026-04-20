import shutil
import uuid
from pathlib import Path

import numpy as np

from analysis.config import EEGConfig, ModelConfig
from analysis.eeg_dataset import WindowDatasetBundle
from analysis.splits import make_loso_splits
from analysis.torch_training import run_torch_classification


def _runtime(name: str) -> Path:
    path = Path("tests") / ".runtime" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _small_config(root: Path) -> EEGConfig:
    return EEGConfig(
        output_dir=root / "runs",
        models=ModelConfig(deep_epochs=1, deep_batch_size=2, deep_learning_rate=0.001, deep_patience=1),
    )


def _small_bundle() -> WindowDatasetBundle:
    labels = []
    subjects = []
    trial_ids = []
    order = []
    rng = np.random.default_rng(42)
    for subject in ["sub1", "sub2"]:
        for class_index, label in enumerate(["negative", "neutral", "positive"]):
            for window_order in range(2):
                labels.append(label)
                subjects.append(subject)
                trial_ids.append(f"{subject}_trial_{class_index}")
                order.append(window_order)
    return WindowDatasetBundle(
        windows=rng.normal(size=(12, 32, 64)).astype("float32"),
        labels=labels,
        subjects=subjects,
        trial_ids=trial_ids,
        window_order=order,
        sfreq=200,
        channel_names=[f"ch{index}" for index in range(32)],
        band_names=["delta", "theta", "alpha", "beta", "gamma"],
    )


def test_supervised_torch_smoke_eegnet() -> None:
    runtime = _runtime("torch_raw")
    bundle = _small_bundle()
    config = _small_config(runtime)

    try:
        metrics = run_torch_classification(
            bundle,
            make_loso_splits(bundle.subjects),
            config,
            runtime / "run_raw",
            deep_network="EEGNet",
            protocol="supervised",
            input_kind="raw",
            device="cpu",
        )

        assert any(item.get("model") == "EEGNet" and "balanced_accuracy" in item for item in metrics)
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


def test_transductive_da_does_not_mark_test_labels_as_training_data() -> None:
    runtime = _runtime("torch_da")
    bundle = _small_bundle()
    config = _small_config(runtime)

    try:
        metrics = run_torch_classification(
            bundle,
            make_loso_splits(bundle.subjects),
            config,
            runtime / "run_da",
            deep_network="BiDANN",
            protocol="transductive_da",
            input_kind="de_sequence",
            sequence_length=2,
            device="cpu",
        )

        scored = [item for item in metrics if "balanced_accuracy" in item]
        assert scored
        assert all(item["uses_test_x_unlabeled"] is True for item in scored)
        assert all(item["uses_test_y_for_training"] is False for item in scored)
    finally:
        shutil.rmtree(runtime, ignore_errors=True)

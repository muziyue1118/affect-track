import shutil
import uuid
from pathlib import Path

import numpy as np

from analysis.config import EEGConfig
from analysis.eeg_dataset import FeatureCache, WindowDatasetBundle, make_model_input
from analysis.features import normalize_windows_per_subject_channel


def test_normalize_windows_per_subject_channel() -> None:
    windows = np.array(
        [
            [[1, 2, 3], [10, 20, 30]],
            [[4, 5, 6], [40, 50, 60]],
            [[101, 102, 103], [1010, 1020, 1030]],
            [[104, 105, 106], [1040, 1050, 1060]],
        ],
        dtype="float32",
    )
    subjects = ["sub1", "sub1", "sub2", "sub2"]

    normalized = normalize_windows_per_subject_channel(windows, subjects)

    for subject in sorted(set(subjects)):
        data = normalized[[index for index, value in enumerate(subjects) if value == subject]]
        assert np.allclose(data.mean(axis=(0, 2)), 0.0, atol=1e-6)
        assert np.allclose(data.std(axis=(0, 2)), 1.0, atol=1e-6)


def test_de_sequence_never_crosses_trial_boundary() -> None:
    runtime = Path("tests") / ".runtime" / f"sequence_{uuid.uuid4().hex}"
    runtime.mkdir(parents=True, exist_ok=True)
    bundle = WindowDatasetBundle(
        windows=np.random.randn(8, 4, 64).astype("float32"),
        labels=["a"] * 4 + ["b"] * 4,
        subjects=["sub1"] * 8,
        trial_ids=["t1"] * 4 + ["t2"] * 4,
        window_order=[0, 1, 2, 3, 0, 1, 2, 3],
        sfreq=64,
        channel_names=["c1", "c2", "c3", "c4"],
        band_names=["delta", "theta", "alpha", "beta", "gamma"],
    )
    try:
        config = EEGConfig(output_dir=runtime / "runs")
        model_input = make_model_input(
            bundle,
            input_kind="de_sequence",
            feature_cache=FeatureCache(config, runtime / "run"),
            sequence_length=3,
            sequence_stride=1,
        )

        assert model_input.x.shape[:3] == (4, 3, 4)
        assert all(
            len({bundle.trial_ids[index] for index in indices}) == 1
            for indices in model_input.source_window_indices
        )
    finally:
        shutil.rmtree(runtime, ignore_errors=True)

import numpy as np
import pytest

from analysis.config import EEGConfig
from analysis.online_preprocessing import (
    crop_filter_context_to_model_window,
    normalize_window_zscore,
    preprocess_online_eeg_window,
    probability_to_score,
    rating_to_online_binary,
)


def test_rating_to_online_binary_drops_neutral() -> None:
    assert rating_to_online_binary(1) == 0
    assert rating_to_online_binary(2) == 0
    assert rating_to_online_binary(3) is None
    assert rating_to_online_binary(4) == 1
    assert rating_to_online_binary(5) == 1


def test_probability_to_score_maps_unit_interval_to_1_5() -> None:
    assert probability_to_score(0.0) == 1.0
    assert probability_to_score(0.5) == 3.0
    assert probability_to_score(1.0) == 5.0
    assert probability_to_score(-1.0) == 1.0
    assert probability_to_score(2.0) == 5.0


def test_window_zscore_normalizes_each_channel() -> None:
    window = np.vstack([np.arange(10), np.arange(10) * 2 + 3]).astype("float32")
    normalized = normalize_window_zscore(window)
    assert normalized.shape == window.shape
    assert np.allclose(normalized.mean(axis=1), 0, atol=1e-6)
    assert np.allclose(normalized.std(axis=1), 1, atol=1e-6)


def test_preprocess_online_eeg_window_outputs_raw_model_shape() -> None:
    pytest.importorskip("scipy")
    config = EEGConfig()
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 10.0, size=(32, 6000)).astype("float32")

    processed = preprocess_online_eeg_window(raw, input_sfreq=1000, config=config, expected_channels=32)

    assert processed.shape == (32, 800)
    assert np.isfinite(processed).all()
    assert np.allclose(processed.mean(axis=1), 0, atol=1e-5)


def test_preprocess_online_eeg_window_rejects_short_filter_context() -> None:
    pytest.importorskip("scipy")
    config = EEGConfig()
    raw = np.zeros((32, 4000), dtype="float32")

    with pytest.raises(ValueError, match="too short for filter context"):
        preprocess_online_eeg_window(raw, input_sfreq=1000, config=config, expected_channels=32)


def test_crop_filter_context_to_model_window_keeps_middle_four_seconds() -> None:
    data = np.arange(1200, dtype="float32")[None, :]

    cropped = crop_filter_context_to_model_window(
        data,
        sfreq=200,
        model_window_seconds=4,
        filter_trim_seconds=1,
    )

    assert cropped.shape == (1, 800)
    assert np.array_equal(cropped[0], np.arange(200, 1000, dtype="float32"))

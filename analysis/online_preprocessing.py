from __future__ import annotations

from fractions import Fraction

from analysis.config import EEGConfig


def normalize_window_zscore(window, eps: float = 1e-6):
    """Normalize one EEG window per channel, matching online deployment."""
    import numpy as np

    data = np.asarray(window, dtype="float32")
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return ((data - mean) / np.maximum(std, eps)).astype("float32", copy=False)


def normalize_windows_zscore(windows):
    import numpy as np

    data = np.asarray(windows, dtype="float32")
    if data.ndim != 3:
        raise ValueError(f"Expected windows shaped (N, C, T), got {data.shape}")
    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)
    return ((data - mean) / np.maximum(std, 1e-6)).astype("float32", copy=False)


def probability_to_score(probability: float) -> float:
    value = 1.0 + 4.0 * float(probability)
    return max(1.0, min(5.0, value))


def rating_to_online_binary(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 2:
        return 0
    if value >= 4:
        return 1
    return None


def preprocess_online_eeg_window(
    raw_window,
    *,
    input_sfreq: float,
    config: EEGConfig,
    expected_channels: int = 32,
):
    """Convert a live EEG buffer to the raw-model deployment tensor.

    The offline training data has already gone through MNE filtering, resampling,
    and average reference. This function mirrors that contract for the online
    Neuracle stream, then applies the deployment-only per-window z-score.
    """
    import numpy as np
    from scipy import signal

    data = np.asarray(raw_window, dtype="float32")
    if data.ndim != 2:
        raise ValueError(f"Expected live EEG shaped (channels, samples), got {data.shape}")
    if data.shape[0] < expected_channels:
        raise ValueError(f"Expected at least {expected_channels} EEG channels, got {data.shape[0]}")
    if data.shape[0] > expected_channels:
        data = data[:expected_channels]
    if not np.isfinite(data).all():
        raise ValueError("Live EEG window contains NaN or Inf")

    threshold = amplitude_threshold(data, config.preprocessing.reject_amplitude_uv)
    if float(np.max(np.abs(data))) > threshold:
        raise ValueError("Live EEG window rejected by amplitude threshold")

    if config.preprocessing.reference == "average":
        data = data - data.mean(axis=0, keepdims=True)

    notch_hz = float(config.preprocessing.notch_hz)
    if notch_hz > 0 and notch_hz < input_sfreq / 2:
        b, a = signal.iirnotch(w0=notch_hz, Q=30.0, fs=input_sfreq)
        data = signal.filtfilt(b, a, data, axis=1).astype("float32", copy=False)

    low, high = config.preprocessing.bandpass_hz
    high = min(float(high), input_sfreq / 2 - 1.0)
    if low > 0 and high > low:
        sos = signal.butter(4, [float(low), float(high)], btype="bandpass", fs=input_sfreq, output="sos")
        data = signal.sosfiltfilt(sos, data, axis=1).astype("float32", copy=False)

    target_sfreq = float(config.preprocessing.resample_hz)
    if abs(float(input_sfreq) - target_sfreq) > 1e-6:
        ratio = Fraction(target_sfreq / float(input_sfreq)).limit_denominator(1000)
        data = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=1).astype("float32", copy=False)

    expected_samples = int(round(float(config.segmentation.window_s) * target_sfreq))
    if data.shape[1] < expected_samples:
        raise ValueError(f"Live EEG window too short after resampling: {data.shape[1]} < {expected_samples}")
    if data.shape[1] > expected_samples:
        data = data[:, -expected_samples:]

    return normalize_window_zscore(data)


def amplitude_threshold(data, reject_amplitude_uv: float) -> float:
    """Use the same volt/microvolt heuristic as offline trial extraction."""
    import numpy as np

    p99 = float(np.nanpercentile(np.abs(data), 99))
    if p99 > 1e-3:
        return float(reject_amplitude_uv)
    return float(reject_amplitude_uv) * 1e-6

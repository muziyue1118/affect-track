from __future__ import annotations

from typing import Sequence

from analysis.config import FeatureConfig


def normalize_windows_per_subject_channel(windows, subjects: Sequence[str]):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for subject/channel normalization. Install environment_eeg.yml first.") from exc

    data = np.asarray(windows, dtype="float32")
    if data.ndim != 3:
        raise ValueError("Expected windows with shape: n_windows x n_channels x n_samples")
    if len(subjects) != data.shape[0]:
        raise ValueError("subjects length must match number of windows")
    normalized = data.copy()
    for subject in sorted(set(subjects)):
        indices = [index for index, value in enumerate(subjects) if value == subject]
        subject_data = data[indices]
        channel_mean = subject_data.mean(axis=(0, 2), keepdims=True)
        channel_std = subject_data.std(axis=(0, 2), keepdims=True)
        channel_std[channel_std < 1e-8] = 1.0
        normalized[indices] = (subject_data - channel_mean) / channel_std
    return normalized


def extract_psd_features(windows, sfreq: float, config: FeatureConfig):
    tensor, band_names = extract_psd_tensor(windows, sfreq=sfreq, config=config)
    return tensor.reshape(tensor.shape[0], -1), band_names


def extract_psd_tensor(windows, sfreq: float, config: FeatureConfig):
    try:
        import numpy as np
        from scipy.signal import welch
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy and SciPy are required for PSD features. Install environment_eeg.yml first.") from exc

    features = []
    band_names = list(config.bands)
    for window in windows:
        freqs, power = welch(window, fs=sfreq, nperseg=min(window.shape[-1], int(2 * sfreq)), axis=-1)
        window_features = []
        for name in band_names:
            low, high = config.bands[name]
            mask = (freqs >= low) & (freqs < high)
            band_power = np.mean(power[:, mask], axis=-1) if mask.any() else np.zeros(window.shape[0])
            window_features.append(np.log(band_power + 1e-12))
        features.append(np.stack(window_features, axis=-1))
    return np.asarray(features, dtype="float32"), band_names


def extract_de_features(windows, sfreq: float, config: FeatureConfig):
    tensor, band_names = extract_de_tensor(windows, sfreq=sfreq, config=config)
    return tensor.reshape(tensor.shape[0], -1), band_names


def extract_de_tensor(windows, sfreq: float, config: FeatureConfig):
    try:
        import numpy as np
        from scipy.signal import butter, sosfiltfilt
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy and SciPy are required for DE features. Install environment_eeg.yml first.") from exc

    features = []
    band_names = list(config.bands)
    nyquist = sfreq / 2.0
    for window in windows:
        window_features = []
        for name in band_names:
            low, high = config.bands[name]
            high = min(high, nyquist - 0.1)
            sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
            filtered = sosfiltfilt(sos, window, axis=-1)
            variance = np.var(filtered, axis=-1, ddof=1)
            de = 0.5 * np.log(2.0 * np.pi * np.e * variance + 1e-12)
            window_features.append(de)
        features.append(np.stack(window_features, axis=-1))
    return np.asarray(features, dtype="float32"), band_names

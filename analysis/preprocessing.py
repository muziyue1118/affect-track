from __future__ import annotations

import os
from pathlib import Path

from analysis.config import EEGConfig


def preprocess_bdf_to_raw(bdf_path: str | Path, config: EEGConfig):
    os.environ.setdefault("MPLCONFIGDIR", str(config.output_dir.parent / ".matplotlib"))
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise RuntimeError("MNE is required for EEG preprocessing. Install environment_eeg.yml first.") from exc

    raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")
    raw.pick("eeg", exclude=[])
    raw.set_montage(config.preprocessing.montage, on_missing="ignore", match_case=False, verbose="ERROR")
    raw.notch_filter(freqs=[config.preprocessing.notch_hz], verbose="ERROR")
    low, high = config.preprocessing.bandpass_hz
    raw.filter(l_freq=low, h_freq=high, verbose="ERROR")
    raw.resample(config.preprocessing.resample_hz, npad="auto", verbose="ERROR")
    if config.preprocessing.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def extract_trial_windows(raw, trial, config: EEGConfig):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for window extraction. Install environment_eeg.yml first.") from exc

    if trial.effective_start_s is None or trial.effective_end_s is None:
        return []
    data = raw.copy().crop(tmin=trial.effective_start_s, tmax=trial.effective_end_s, include_tmax=False).get_data()
    sfreq = float(raw.info["sfreq"])
    window_samples = int(round(config.segmentation.window_s * sfreq))
    overlap_samples = int(round(config.segmentation.window_overlap_s * sfreq))
    step = window_samples - overlap_samples
    if window_samples <= 0 or step <= 0:
        raise ValueError("window_s must be positive and larger than window_overlap_s")
    threshold = _amplitude_threshold(data, config.preprocessing.reject_amplitude_uv)
    windows = []
    for start in range(0, max(0, data.shape[1] - window_samples + 1), step):
        window = data[:, start : start + window_samples]
        if window.shape[1] != window_samples:
            continue
        if not np.isfinite(window).all():
            continue
        if np.max(np.abs(window)) > threshold:
            continue
        windows.append(window.astype("float32", copy=False))
    return windows


def _amplitude_threshold(data, reject_amplitude_uv: float) -> float:
    """MNE returns volts for standard BDF units, but some devices expose microvolt-like units."""
    import numpy as np

    p99 = float(np.nanpercentile(np.abs(data), 99))
    if p99 > 1e-3:
        return reject_amplitude_uv
    return reject_amplitude_uv * 1e-6

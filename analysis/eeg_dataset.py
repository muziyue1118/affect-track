from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from analysis.config import EEGConfig
from analysis.features import extract_de_tensor, extract_psd_tensor, normalize_windows_per_subject_channel


@dataclass(frozen=True)
class WindowDatasetBundle:
    windows: object
    labels: list[str]
    subjects: list[str]
    trial_ids: list[str]
    window_order: list[int]
    sfreq: float
    channel_names: list[str]
    band_names: list[str]


@dataclass(frozen=True)
class ModelInputBundle:
    x: object
    labels: list[str]
    subjects: list[str]
    trial_ids: list[str]
    source_window_indices: list[tuple[int, ...]]
    input_kind: str
    band_names: list[str]


class FeatureCache:
    def __init__(self, config: EEGConfig, run_dir: Path):
        self.config = config
        self.cache_dir = config.output_dir.parent / "eeg_cache" / self._cache_key()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._memory: dict[str, object] = {}

    def _cache_key(self) -> str:
        payload = {
            "preprocessing": self.config.preprocessing.__dict__,
            "segmentation": self.config.segmentation.__dict__,
            "features": self.config.features.__dict__,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

    def de_tensor(self, bundle: WindowDatasetBundle):
        if "de" in self._memory:
            return self._memory["de"]
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("NumPy is required for DE feature caching.") from exc
        path = self.cache_dir / "de_tensor.npz"
        metadata = self._bundle_metadata(bundle)
        normalized = normalize_windows_per_subject_channel(bundle.windows, bundle.subjects)
        if path.exists():
            payload = np.load(path, allow_pickle=True)
            tensor = payload["tensor"]
            band_names = payload["band_names"].tolist()
            cached_metadata = str(payload["metadata"].tolist()) if "metadata" in payload else ""
            if tensor.shape[0] == len(bundle.labels) and cached_metadata == metadata:
                self._memory["de"] = (tensor, band_names)
                return self._memory["de"]
        tensor, band_names = extract_de_tensor(normalized, sfreq=bundle.sfreq, config=self.config.features)
        np.savez_compressed(path, tensor=tensor, band_names=np.asarray(band_names), metadata=np.asarray(metadata))
        self._memory["de"] = (tensor, band_names)
        return self._memory["de"]

    def psd_tensor(self, bundle: WindowDatasetBundle):
        if "psd" in self._memory:
            return self._memory["psd"]
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("NumPy is required for PSD feature caching.") from exc
        path = self.cache_dir / "psd_tensor.npz"
        metadata = self._bundle_metadata(bundle)
        normalized = normalize_windows_per_subject_channel(bundle.windows, bundle.subjects)
        if path.exists():
            payload = np.load(path, allow_pickle=True)
            tensor = payload["tensor"]
            band_names = payload["band_names"].tolist()
            cached_metadata = str(payload["metadata"].tolist()) if "metadata" in payload else ""
            if tensor.shape[0] == len(bundle.labels) and cached_metadata == metadata:
                self._memory["psd"] = (tensor, band_names)
                return self._memory["psd"]
        tensor, band_names = extract_psd_tensor(normalized, sfreq=bundle.sfreq, config=self.config.features)
        np.savez_compressed(path, tensor=tensor, band_names=np.asarray(band_names), metadata=np.asarray(metadata))
        self._memory["psd"] = (tensor, band_names)
        return self._memory["psd"]

    def _bundle_metadata(self, bundle: WindowDatasetBundle) -> str:
        payload = {
            "n_windows": len(bundle.labels),
            "labels": bundle.labels,
            "subjects": bundle.subjects,
            "trial_ids": bundle.trial_ids,
            "window_order": bundle.window_order,
            "sfreq": bundle.sfreq,
            "shape": getattr(bundle.windows, "shape", None),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def make_model_input(
    bundle: WindowDatasetBundle,
    *,
    input_kind: str,
    feature_cache: FeatureCache,
    sequence_length: int = 9,
    sequence_stride: int = 1,
) -> ModelInputBundle:
    if input_kind == "raw":
        return ModelInputBundle(
            x=bundle.windows,
            labels=bundle.labels,
            subjects=bundle.subjects,
            trial_ids=bundle.trial_ids,
            source_window_indices=[(index,) for index in range(len(bundle.labels))],
            input_kind="raw",
            band_names=bundle.band_names,
        )
    if input_kind == "de":
        tensor, band_names = feature_cache.de_tensor(bundle)
        return ModelInputBundle(
            x=tensor,
            labels=bundle.labels,
            subjects=bundle.subjects,
            trial_ids=bundle.trial_ids,
            source_window_indices=[(index,) for index in range(len(bundle.labels))],
            input_kind="de",
            band_names=list(band_names),
        )
    if input_kind == "de_sequence":
        tensor, band_names = feature_cache.de_tensor(bundle)
        return _make_sequence_bundle(
            tensor,
            bundle.labels,
            bundle.subjects,
            bundle.trial_ids,
            bundle.window_order,
            band_names=list(band_names),
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
        )
    raise ValueError(f"Unsupported input kind: {input_kind}")


def _make_sequence_bundle(
    tensor,
    labels: Sequence[str],
    subjects: Sequence[str],
    trial_ids: Sequence[str],
    window_order: Sequence[int],
    *,
    band_names: list[str],
    sequence_length: int,
    sequence_stride: int,
) -> ModelInputBundle:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for sequence input construction.") from exc
    if sequence_length < 1 or sequence_stride < 1:
        raise ValueError("sequence_length and sequence_stride must be positive integers")
    sequences = []
    seq_labels: list[str] = []
    seq_subjects: list[str] = []
    seq_trials: list[str] = []
    source_indices: list[tuple[int, ...]] = []
    by_trial: dict[str, list[int]] = {}
    for index, trial_id in enumerate(trial_ids):
        by_trial.setdefault(trial_id, []).append(index)
    for trial_id, indices in by_trial.items():
        ordered = sorted(indices, key=lambda index: window_order[index])
        if len(ordered) < sequence_length:
            continue
        for start in range(0, len(ordered) - sequence_length + 1, sequence_stride):
            selected = tuple(ordered[start : start + sequence_length])
            trial_labels = {labels[index] for index in selected}
            trial_subjects = {subjects[index] for index in selected}
            if len(trial_labels) != 1 or len(trial_subjects) != 1:
                raise ValueError(f"Invalid sequence crossing label/subject boundary in trial {trial_id}")
            sequences.append(tensor[list(selected)])
            seq_labels.append(labels[selected[-1]])
            seq_subjects.append(subjects[selected[-1]])
            seq_trials.append(trial_id)
            source_indices.append(selected)
    return ModelInputBundle(
        x=np.asarray(sequences, dtype="float32"),
        labels=seq_labels,
        subjects=seq_subjects,
        trial_ids=seq_trials,
        source_window_indices=source_indices,
        input_kind="de_sequence",
        band_names=band_names,
    )

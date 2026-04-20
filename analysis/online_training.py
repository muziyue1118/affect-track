from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.Net import DEEP_MODEL_NAMES, get_model_spec, build_torch_model
from analysis.audit import AuditResult, TrialRecord, run_audit
from analysis.config import EEGConfig, load_config
from analysis.eeg_dataset import WindowDatasetBundle
from analysis.eeg_pipeline import _label_for_task, filter_audit_result_by_subjects, parse_subject_key_filters
from analysis.online_preprocessing import preprocess_online_eeg_window, probability_to_score, rating_to_online_binary
from analysis.time_utils import format_run_timestamp


ONLINE_TASKS = {
    "valence": {
        "pipeline_task": "valence_binary",
        "low_label": "negative",
        "high_label": "positive",
        "rating_field": "valence",
    },
    "arousal": {
        "pipeline_task": "arousal_binary",
        "low_label": "low",
        "high_label": "high",
        "rating_field": "arousal",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train deployment models for online Valence/Arousal prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train valence and arousal deployment models.")
    train_parser.add_argument("--config", default="analysis/eeg_config.yaml")
    train_parser.add_argument("--network", default="FBSTCNet", choices=DEEP_MODEL_NAMES)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--output-dir", default="models/emotion_online")
    train_parser.add_argument("--run-id", default=None)
    train_parser.add_argument("--subject-key", default=None, help="Optional single subject filter, e.g. sub3.")
    train_parser.add_argument("--holdout-category-trials", action="store_true")
    train_parser.add_argument("--test-positive-trials", type=int, default=1)
    train_parser.add_argument("--test-negative-trials", type=int, default=1)
    train_parser.add_argument("--split-seed", type=int, default=42)
    train_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "train":
        train_online_models(
            config,
            network=args.network,
            device=args.device,
            output_dir=Path(args.output_dir),
            run_id=args.run_id,
            subject_key=args.subject_key,
            holdout_category_trials=args.holdout_category_trials,
            test_positive_trials=args.test_positive_trials,
            test_negative_trials=args.test_negative_trials,
            split_seed=args.split_seed,
            overwrite=args.overwrite,
        )
        print(f"Online models written to {Path(args.output_dir)}")
        return 0
    return 1


def train_online_models(
    config: EEGConfig,
    *,
    network: str = "FBSTCNet",
    device: str = "auto",
    output_dir: Path = Path("models/emotion_online"),
    run_id: str | None = None,
    subject_key: str | None = None,
    holdout_category_trials: bool = False,
    test_positive_trials: int = 1,
    test_negative_trials: int = 1,
    split_seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy and PyTorch are required for online deployment training.") from exc

    spec = get_model_spec(network)
    if spec.input_kind != "raw" or spec.protocol != "supervised":
        raise ValueError(f"Online deployment currently requires a supervised raw EEG model, got {network}.")

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.benchmark = True
    device_obj = _resolve_device(device, torch)

    filter_context_seconds = 6.0
    filter_trim_seconds = 1.0
    _guard_output_dir(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = run_audit(config, run_id=run_id or f"online_train_{format_run_timestamp()}")
    if not audit.valid_trials:
        raise RuntimeError(f"No valid aligned EEG trials. See {audit.run_dir / 'report.md'}")
    subject_keys = parse_subject_key_filters(subject_key, None)
    if subject_keys:
        audit = filter_audit_result_by_subjects(audit, subject_keys)
        if not audit.valid_trials:
            raise RuntimeError(f"No valid aligned EEG trials for subjects {', '.join(subject_keys)}.")
    holdout_trials = (
        _select_constrained_category_holdout(
            audit.valid_trials,
            positive_count=test_positive_trials,
            negative_count=test_negative_trials,
            split_seed=split_seed,
        )
        if holdout_category_trials
        else []
    )
    holdout_trial_ids = {trial.trial_id for trial in holdout_trials}

    task_summaries: dict[str, object] = {}
    artifact_paths: dict[str, str] = {}
    for task_name, task_info in ONLINE_TASKS.items():
        bundle = _build_online_window_dataset(
            audit,
            config,
            task_info["pipeline_task"],
            filter_context_seconds=filter_context_seconds,
            filter_trim_seconds=filter_trim_seconds,
        )
        if not bundle.labels:
            raise RuntimeError(f"No windows available after dropping neutral ratings for {task_name}.")

        x_np = bundle.windows
        y_np = np.asarray([1 if label == "high" else 0 for label in bundle.labels], dtype="float32")
        if len(set(y_np.tolist())) < 2:
            raise RuntimeError(f"{task_name} has fewer than two binary classes after filtering.")
        train_indices, test_indices = _split_window_indices_by_trial(bundle.trial_ids, holdout_trial_ids)
        if not train_indices:
            raise RuntimeError(f"No training windows available for {task_name}.")
        if len(set(y_np[train_indices].tolist())) < 2:
            raise RuntimeError(f"{task_name} training split has fewer than two binary classes.")
        if holdout_category_trials:
            if not test_indices:
                raise RuntimeError(f"No test windows available for {task_name}; selected holdout trials were filtered out.")
            if len(set(y_np[test_indices].tolist())) < 2:
                raise RuntimeError(f"{task_name} test split has fewer than two binary classes.")
        x_train, y_train = x_np[train_indices], y_np[train_indices]
        x_test = x_np[test_indices] if test_indices else None
        y_test = y_np[test_indices] if test_indices else None

        model, history = _train_single_binary_model(
            x_train,
            y_train,
            network=network,
            config=config,
            device=device_obj,
            x_test=x_test,
            y_test=y_test,
        )
        train_metrics = _evaluate_binary_model(
            model,
            x_train,
            y_train,
            device=device_obj,
            batch_size=config.models.deep_batch_size,
        )
        test_metrics = (
            _evaluate_binary_model(
                model,
                x_test,
                y_test,
                device=device_obj,
                batch_size=config.models.deep_batch_size,
            )
            if x_test is not None and y_test is not None
            else None
        )
        best_epoch = _best_epoch_from_history(history)
        artifact_name = f"{task_name}_{network.lower()}.pt"
        artifact_path = output_dir / artifact_name
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "task": task_name,
                "network": network,
                "n_channels": int(x_np.shape[1]),
                "n_samples": int(x_np.shape[2]),
                "sfreq": float(config.preprocessing.resample_hz),
                "filter_context_seconds": filter_context_seconds,
                "filter_trim_seconds": filter_trim_seconds,
                "score_formula": "1 + 4 * sigmoid(logit)",
                "low_label": task_info["low_label"],
                "high_label": task_info["high_label"],
                "best_epoch": best_epoch["epoch"] if best_epoch else None,
            },
            artifact_path,
        )
        artifact_paths[task_name] = artifact_name
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        test_positive_count = int(y_test.sum()) if y_test is not None else 0
        test_negative_count = int(len(y_test) - test_positive_count) if y_test is not None else 0
        task_summaries[task_name] = {
            "task": task_name,
            "pipeline_task": task_info["pipeline_task"],
            "network": network,
            "artifact": artifact_name,
            "n_windows": int(len(y_np)),
            "n_trials": int(len(set(bundle.trial_ids))),
            "n_subjects": int(len(set(bundle.subjects))),
            "class_counts": {
                "train": {
                    task_info["low_label"]: negatives,
                    task_info["high_label"]: positives,
                },
                "test": {
                    task_info["low_label"]: test_negative_count,
                    task_info["high_label"]: test_positive_count,
                }
                if y_test is not None
                else None,
            },
            "train_windows": int(len(y_train)),
            "test_windows": int(len(y_test)) if y_test is not None else 0,
            "train_trials": sorted({bundle.trial_ids[index] for index in train_indices}),
            "test_trials": sorted({bundle.trial_ids[index] for index in test_indices}),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
            "final_loss": float(history[-1]["loss"]) if history else None,
            "history": history,
            "example_probability_to_score": {
                "0.0": probability_to_score(0.0),
                "0.5": probability_to_score(0.5),
                "1.0": probability_to_score(1.0),
            },
        }

    metadata = {
        "created_by": "analysis.online_training",
        "network": network,
        "artifacts": artifact_paths,
        "tasks": task_summaries,
        "preprocessing": {
            "input_kind": "raw",
            "normalization": "per_window_per_channel_zscore",
            "filter_context_seconds": filter_context_seconds,
            "filter_trim_seconds": filter_trim_seconds,
            "model_window_seconds": float(config.segmentation.window_s),
            "live_device": {
                "protocol": "Neuracle",
                "channels": 32,
                "srate": 1000,
                "host": "127.0.0.1",
                "port": 8712,
            },
            "offline_config": asdict(config.preprocessing),
            "segmentation": asdict(config.segmentation),
        },
        "training": {
            "uses_all_available_binary_windows": not holdout_category_trials,
            "neutral_rating_3_dropped": True,
            "subject_filter": list(subject_keys),
            "holdout_category_trials": holdout_category_trials,
            "holdout_trials": [
                {
                    "trial_id": trial.trial_id,
                    "video_name": trial.video_name,
                    "category": trial.category,
                    "valence": trial.valence,
                    "arousal": trial.arousal,
                }
                for trial in holdout_trials
            ],
            "split_seed": split_seed,
            "selection_criterion": "highest_test_balanced_accuracy_then_lowest_test_loss_then_earliest_epoch",
            "loss": "BCEWithLogitsLoss",
            "device_requested": device,
            "device_resolved": str(device_obj),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else None,
            "audit_run_dir": str(audit.run_dir),
        },
        "config": asdict(config),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_report(output_dir / "report.md", metadata)
    return metadata


def _train_single_binary_model(
    x_np,
    y_np,
    *,
    network: str,
    config: EEGConfig,
    device,
    x_test=None,
    y_test=None,
):
    import numpy as np
    import torch
    from torch import nn

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    model = build_torch_model(
        network,
        n_channels=int(x.shape[1]),
        n_classes=1,
        n_samples=int(x.shape[2]),
    ).to(device)
    positives = float(y.sum().item())
    negatives = float(len(y) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.models.deep_learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(config.random_seed)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.models.deep_batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=bool(getattr(device, "type", None) == "cuda"),
    )
    history = []
    best_state = None
    best_key = None
    for epoch in range(config.models.deep_epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=bool(getattr(device, "type", None) == "cuda"))
            batch_y = batch_y.to(device, non_blocking=bool(getattr(device, "type", None) == "cuda"))
            optimizer.zero_grad()
            logits = _logits(model(batch_x)).reshape(-1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else None
        epoch_row = {"epoch": epoch + 1, "loss": train_loss}
        if x_test is not None and y_test is not None:
            test_metrics = _evaluate_binary_model(
                model,
                x_test,
                y_test,
                device=device,
                batch_size=config.models.deep_batch_size,
                pos_weight=float(pos_weight.detach().cpu().item()),
            )
            epoch_row.update(
                {
                    "test_loss": test_metrics.get("loss"),
                    "test_accuracy": test_metrics["accuracy"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                }
            )
            key = (
                float(test_metrics["balanced_accuracy"]),
                -float(test_metrics.get("loss") or 0.0),
                -(epoch + 1),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())
        history.append(epoch_row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _evaluate_binary_model(model, x_np, y_np, *, device, batch_size: int, pos_weight: float | None = None) -> dict[str, object]:
    import numpy as np
    import torch

    x = torch.tensor(x_np, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)
    probabilities: list[float] = []
    logits_all: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device, non_blocking=bool(getattr(device, "type", None) == "cuda"))
            logits = _logits(model(batch_x)).reshape(-1)
            logits_all.extend(logits.detach().cpu().numpy().astype(float).tolist())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
    metrics = _binary_metrics(np.asarray(y_np, dtype="int64"), np.asarray(probabilities, dtype="float64"))
    if pos_weight is not None:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        loss = loss_fn(
            torch.tensor(logits_all, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32),
        )
        metrics["loss"] = float(loss.item())
    return metrics


def _binary_metrics(y_true, probabilities) -> dict[str, object]:
    import numpy as np

    y_true = np.asarray(y_true, dtype="int64")
    probabilities = np.asarray(probabilities, dtype="float64")
    y_pred = (probabilities >= 0.5).astype("int64")
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = int(len(y_true))
    accuracy = float((tp + tn) / total) if total else 0.0
    recall_low = float(tn / (tn + fp)) if (tn + fp) else 0.0
    recall_high = float(tp / (tp + fn)) if (tp + fn) else 0.0
    balanced_accuracy = (recall_low + recall_high) / 2.0
    f1_low = float((2 * tn) / (2 * tn + fn + fp)) if (2 * tn + fn + fp) else 0.0
    f1_high = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": (f1_low + f1_high) / 2.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "prediction_counts": {
            "low": int((y_pred == 0).sum()),
            "high": int((y_pred == 1).sum()),
        },
        "probability_summary": {
            "min": float(probabilities.min()) if total else None,
            "mean": float(probabilities.mean()) if total else None,
            "max": float(probabilities.max()) if total else None,
        },
    }


def _logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output


def _resolve_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA in this environment.")
    return resolved


def _guard_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    protected_patterns = ("metadata.json", "report.md", "valence_*.pt", "arousal_*.pt")
    if overwrite or not output_dir.exists():
        return
    protected = []
    for pattern in protected_patterns:
        protected.extend(output_dir.glob(pattern))
    if protected:
        names = ", ".join(sorted(path.name for path in protected[:5]))
        raise FileExistsError(
            f"Output directory already contains online model artifacts: {output_dir} ({names}). "
            "Use --overwrite or choose a new --output-dir."
        )


def _build_online_window_dataset(
    result: AuditResult,
    config: EEGConfig,
    task: str,
    *,
    filter_context_seconds: float,
    filter_trim_seconds: float,
) -> WindowDatasetBundle:
    import numpy as np

    expected_context = float(config.segmentation.window_s) + 2.0 * float(filter_trim_seconds)
    if abs(float(filter_context_seconds) - expected_context) > 1e-6:
        raise ValueError(
            f"filter_context_seconds must equal window_s + 2 * filter_trim_seconds "
            f"({expected_context}), got {filter_context_seconds}"
        )

    headers_by_subject = {row["subject_key"]: Path(str(row["bdf_path"])) for row in result.subject_rows}
    raw_cache = {}
    windows = []
    labels = []
    subjects = []
    trial_ids = []
    window_order = []
    channel_names: list[str] = []
    for trial in result.valid_trials:
        label = _label_for_task(trial, task)
        if label is None:
            continue
        bdf_path = headers_by_subject[trial.subject_key]
        if trial.subject_key not in raw_cache:
            raw_cache[trial.subject_key] = _load_raw_bdf_for_online_training(bdf_path, config)
        raw = raw_cache[trial.subject_key]
        if not channel_names:
            channel_names = list(raw.ch_names[:32])
        trial_windows = _extract_online_context_windows(
            raw,
            trial,
            config,
            filter_trim_seconds=filter_trim_seconds,
        )
        for order, window in enumerate(trial_windows):
            windows.append(window)
            labels.append(label)
            subjects.append(trial.subject_key)
            trial_ids.append(trial.trial_id)
            window_order.append(order)
    return WindowDatasetBundle(
        windows=np.asarray(windows, dtype="float32"),
        labels=labels,
        subjects=subjects,
        trial_ids=trial_ids,
        window_order=window_order,
        sfreq=float(config.preprocessing.resample_hz),
        channel_names=channel_names,
        band_names=list(config.features.bands),
    )


def _load_raw_bdf_for_online_training(bdf_path: Path, config: EEGConfig):
    os.environ.setdefault("MPLCONFIGDIR", str(config.output_dir.parent / ".matplotlib"))
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise RuntimeError("MNE is required for online deployment training.") from exc

    raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")
    raw.pick("eeg", exclude=[])
    raw.set_montage(config.preprocessing.montage, on_missing="ignore", match_case=False, verbose="ERROR")
    return raw


def _extract_online_context_windows(raw, trial: TrialRecord, config: EEGConfig, *, filter_trim_seconds: float):
    import numpy as np

    if trial.effective_start_s is None or trial.effective_end_s is None:
        return []
    sfreq = float(raw.info["sfreq"])
    model_window_s = float(config.segmentation.window_s)
    overlap_s = float(config.segmentation.window_overlap_s)
    step_s = model_window_s - overlap_s
    if model_window_s <= 0 or step_s <= 0:
        raise ValueError("window_s must be positive and larger than window_overlap_s")

    windows = []
    last_target_start = float(trial.effective_end_s) - model_window_s
    target_start = float(trial.effective_start_s)
    while target_start <= last_target_start + 1e-9:
        context_start = target_start - float(filter_trim_seconds)
        context_end = target_start + model_window_s + float(filter_trim_seconds)
        if context_start >= 0 and context_end <= float(raw.times[-1]):
            start_sample, stop_sample = raw.time_as_index([context_start, context_end], use_rounding=True)
            context = raw.get_data(start=int(start_sample), stop=int(stop_sample))
            try:
                processed = preprocess_online_eeg_window(
                    context,
                    input_sfreq=sfreq,
                    config=config,
                    expected_channels=32,
                    filter_trim_seconds=filter_trim_seconds,
                )
            except ValueError:
                target_start += step_s
                continue
            windows.append(processed.astype("float32", copy=False))
        target_start += step_s
    return windows


def _select_constrained_category_holdout(
    valid_trials: Sequence[TrialRecord],
    *,
    positive_count: int,
    negative_count: int,
    split_seed: int,
) -> list[TrialRecord]:
    if positive_count < 0 or negative_count < 0:
        raise ValueError("test-positive-trials and test-negative-trials must be non-negative")
    positives = sorted([trial for trial in valid_trials if trial.category == "positive"], key=lambda trial: trial.video_name)
    negatives = sorted([trial for trial in valid_trials if trial.category == "negative"], key=lambda trial: trial.video_name)
    if len(positives) < positive_count or len(negatives) < negative_count:
        raise RuntimeError(
            f"Not enough category trials for holdout: positive {len(positives)}/{positive_count}, "
            f"negative {len(negatives)}/{negative_count}."
        )
    import itertools

    candidates: list[tuple[TrialRecord, ...]] = []
    for pos_combo in itertools.combinations(positives, positive_count):
        for neg_combo in itertools.combinations(negatives, negative_count):
            selected = tuple(sorted((*pos_combo, *neg_combo), key=lambda trial: trial.trial_id))
            if _holdout_has_binary_coverage(valid_trials, selected):
                candidates.append(selected)
    if not candidates:
        raise RuntimeError(
            "No positive/negative holdout pair can provide low/high coverage for both valence and arousal "
            "while leaving both classes in training."
        )
    rng = random.Random(split_seed)
    return list(rng.choice(candidates))


def _holdout_has_binary_coverage(all_trials: Sequence[TrialRecord], holdout: Sequence[TrialRecord]) -> bool:
    holdout_ids = {trial.trial_id for trial in holdout}
    train = [trial for trial in all_trials if trial.trial_id not in holdout_ids]
    for field in ("valence", "arousal"):
        holdout_labels = {
            label
            for trial in holdout
            for label in [rating_to_online_binary(getattr(trial, field))]
            if label is not None
        }
        train_labels = {
            label
            for trial in train
            for label in [rating_to_online_binary(getattr(trial, field))]
            if label is not None
        }
        if holdout_labels != {0, 1} or train_labels != {0, 1}:
            return False
    return True


def _split_window_indices_by_trial(trial_ids: Sequence[str], test_trial_ids: set[str]) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    test_indices: list[int] = []
    for index, trial_id in enumerate(trial_ids):
        if trial_id in test_trial_ids:
            test_indices.append(index)
        else:
            train_indices.append(index)
    return train_indices, test_indices


def _best_epoch_from_history(history: Sequence[dict[str, object]]) -> dict[str, object] | None:
    rows = [row for row in history if row.get("test_balanced_accuracy") is not None]
    if not rows:
        return history[-1] if history else None
    return max(
        rows,
        key=lambda row: (
            float(row.get("test_balanced_accuracy") or 0.0),
            -float(row.get("test_loss") or 0.0),
            -int(row.get("epoch") or 0),
        ),
    )


def _write_report(path: Path, metadata: dict[str, object]) -> None:
    lines = [
        "# Online Valence/Arousal Deployment Training",
        "",
        f"- Network: {metadata['network']}",
        f"- Normalization: {metadata['preprocessing']['normalization']}",
        (
            f"- Filter context: {metadata['preprocessing']['filter_context_seconds']}s, "
            f"trim {metadata['preprocessing']['filter_trim_seconds']}s each side, "
            f"model window {metadata['preprocessing']['model_window_seconds']}s"
        ),
        f"- Device: {metadata['training']['device_resolved']}",
        "- Final artifacts keep the best epoch by test balanced accuracy when a holdout split is enabled.",
        "- If holdout is enabled, test metrics are used for model selection and are therefore deployment-tuning metrics.",
        f"- Subject filter: {metadata['training']['subject_filter'] or 'all'}",
        f"- Holdout trials: {metadata['training']['holdout_trials'] or 'none'}",
        "",
        "## Tasks",
        "",
    ]
    for task_name, summary in metadata["tasks"].items():
        counts = summary["class_counts"]
        test_metrics = summary.get("test_metrics")
        best_epoch = summary.get("best_epoch") or {}
        lines.extend(
            [
                f"### {task_name}",
                "",
                f"- Artifact: {summary['artifact']}",
                f"- Windows: train {summary['train_windows']}, test {summary['test_windows']}",
                f"- Trials: train {len(summary['train_trials'])}, test {len(summary['test_trials'])}",
                f"- Subjects: {summary['n_subjects']}",
                f"- Class counts: {counts}",
                f"- Train accuracy: {summary['train_metrics']['accuracy']:.4f}",
                f"- Train balanced accuracy: {summary['train_metrics']['balanced_accuracy']:.4f}",
                f"- Train macro F1: {summary['train_metrics']['macro_f1']:.4f}",
                f"- Train confusion matrix [[TN, FP], [FN, TP]]: {summary['train_metrics']['confusion_matrix']}",
                (
                    f"- Test accuracy: {test_metrics['accuracy']:.4f}"
                    if test_metrics
                    else "- Test accuracy: n/a"
                ),
                (
                    f"- Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}"
                    if test_metrics
                    else "- Test balanced accuracy: n/a"
                ),
                (
                    f"- Test macro F1: {test_metrics['macro_f1']:.4f}"
                    if test_metrics
                    else "- Test macro F1: n/a"
                ),
                (
                    f"- Test confusion matrix [[TN, FP], [FN, TP]]: {test_metrics['confusion_matrix']}"
                    if test_metrics
                    else "- Test confusion matrix: n/a"
                ),
                f"- Best epoch: {best_epoch.get('epoch')}",
                f"- Best test loss: {best_epoch.get('test_loss')}",
                f"- Best test balanced accuracy: {best_epoch.get('test_balanced_accuracy')}",
                f"- Final loss: {summary['final_loss']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

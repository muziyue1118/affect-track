from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit import AuditResult, run_audit
from analysis.classical_models import run_feature_classification
from analysis.config import EEGConfig, load_config
from analysis.eeg_dataset import FeatureCache, WindowDatasetBundle
from analysis.Net import DEEP_MODEL_NAMES
from analysis.labels import normalize_subject_id
from analysis.preprocessing import extract_trial_windows, preprocess_bdf_to_raw
from analysis.splits import make_loso_splits, make_subject_dependent_splits, make_window_kfold_splits
from analysis.time_utils import format_run_timestamp
from analysis.torch_training import run_torch_classification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and classify AffectTrack EEG recordings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit EEG files and label alignment.")
    audit_parser.add_argument("--config", default="analysis/eeg_config.yaml")
    audit_parser.add_argument("--run-id", default=None)

    run_parser = subparsers.add_parser("run", help="Run preprocessing, features, and classification.")
    run_parser.add_argument("--config", default="analysis/eeg_config.yaml")
    run_parser.add_argument("--task", default="category", choices=["category", "valence_binary", "arousal_binary"])
    run_parser.add_argument("--split-mode", default=None, choices=["loso", "subject_dependent", "window_kfold"])
    run_parser.add_argument("--model", default="all", choices=["all", "features", "deep"])
    run_parser.add_argument("--feature-kind", default="all", choices=["all", "de", "psd"])
    run_parser.add_argument(
        "--classifier",
        default="all",
        choices=["all", "logistic_regression", "linear_svm", "rbf_svm", "random_forest"],
    )
    run_parser.add_argument("--deep-network", default="shallow_convnet", choices=["all", *DEEP_MODEL_NAMES])
    run_parser.add_argument("--protocol", default="supervised", choices=["supervised", "source_dg", "transductive_da"])
    run_parser.add_argument("--input-kind", default="auto", choices=["auto", "raw", "de", "de_sequence"])
    run_parser.add_argument("--sequence-length", type=int, default=9)
    run_parser.add_argument("--sequence-stride", type=int, default=1)
    run_parser.add_argument("--device", default=None, help="Torch device for deep models: auto, cpu, cuda, cuda:0, ...")
    run_parser.add_argument("--subject-key", default=None, help="Optional single normalized subject filter, e.g. sub3 or 003.")
    run_parser.add_argument(
        "--subject-keys",
        nargs="*",
        default=None,
        help="Optional subject filters, e.g. sub3 sub4 sub5 or sub3,sub4,sub5.",
    )
    run_parser.add_argument("--run-id", default=None)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "audit":
        result = run_audit(config, run_id=args.run_id)
        print(f"Audit written to {result.run_dir}")
        print(f"Valid trials: {len(result.valid_trials)} / {len(result.trial_records)}")
        return 0

    split_mode = args.split_mode or config.evaluation.split_mode
    config = replace(config, evaluation=replace(config.evaluation, split_mode=split_mode))
    result = run_audit(config, run_id=args.run_id or format_run_timestamp())
    if not result.valid_trials:
        print(f"No valid aligned trials. Training refused. See {result.run_dir / 'report.md'}")
        return 2
    subject_keys = parse_subject_key_filters(args.subject_key, args.subject_keys)
    if subject_keys:
        result = filter_audit_result_by_subjects(result, subject_keys)
        if not result.valid_trials:
            print(f"No valid aligned trials for subjects {', '.join(subject_keys)}. See {result.run_dir / 'report.md'}")
            return 2
    try:
        run_classification(
            result,
            config,
            task=args.task,
            model=args.model,
            feature_kind=args.feature_kind,
            classifier=args.classifier,
            deep_network=args.deep_network,
            protocol=args.protocol,
            input_kind=args.input_kind,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            device=args.device or config.models.deep_device,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        if "required" in str(exc).lower():
            print("Create the EEG environment with: conda env create -f environment_eeg.yml")
        return 3
    print(f"Run written to {result.run_dir}")
    return 0


def filter_audit_result_by_subject(result: AuditResult, subject_key: str) -> AuditResult:
    return filter_audit_result_by_subjects(result, [subject_key])


def filter_audit_result_by_subjects(result: AuditResult, subject_keys: list[str] | tuple[str, ...]) -> AuditResult:
    normalized = set(parse_subject_key_filters(None, list(subject_keys)))
    return AuditResult(
        run_dir=result.run_dir,
        subject_rows=[row for row in result.subject_rows if row.get("subject_key") in normalized],
        label_records=[record for record in result.label_records if record.subject_key in normalized],
        trial_records=[record for record in result.trial_records if record.subject_key in normalized],
    )


def parse_subject_key_filters(subject_key: str | None, subject_keys: list[str] | None) -> tuple[str, ...]:
    raw_values: list[str] = []
    if subject_key:
        raw_values.append(subject_key)
    if subject_keys:
        raw_values.extend(subject_keys)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            key = normalize_subject_id(item)
            if key not in seen:
                normalized.append(key)
                seen.add(key)
    return tuple(normalized)


def run_classification(
    result: AuditResult,
    config: EEGConfig,
    task: str,
    model: str,
    feature_kind: str = "all",
    classifier: str = "all",
    deep_network: str = "shallow_convnet",
    protocol: str = "supervised",
    input_kind: str = "auto",
    sequence_length: int = 9,
    sequence_stride: int = 1,
    device: str = "auto",
) -> None:
    bundle = _build_window_dataset(result, config, task)
    if not bundle.labels:
        raise RuntimeError(f"No windows available for task {task}")
    splits = _make_splits(bundle.subjects, bundle.trial_ids, bundle.labels, config)
    if not splits:
        raise RuntimeError(f"No valid splits for mode {config.evaluation.split_mode}")
    metadata_rows = [
        {
            "window_index": index,
            "label": bundle.labels[index],
            "subject": bundle.subjects[index],
            "trial_id": bundle.trial_ids[index],
            "window_order": bundle.window_order[index],
        }
        for index in range(len(bundle.labels))
    ]
    _write_csv(result.run_dir / "window_manifest.csv", metadata_rows)
    metric_sets: list[dict[str, object]] = []
    feature_cache = FeatureCache(config, result.run_dir)
    if model in {"all", "features"}:
        if feature_kind in {"all", "psd"}:
            psd_tensor, _ = feature_cache.psd_tensor(bundle)
            psd = psd_tensor.reshape(psd_tensor.shape[0], -1)
            metric_sets.extend(
                run_feature_classification(
                    psd,
                    bundle.labels,
                    bundle.subjects,
                    bundle.trial_ids,
                    splits,
                    config,
                    result.run_dir,
                    "psd",
                    classifier=classifier,
                )
            )
        if feature_kind in {"all", "de"}:
            de_tensor, _ = feature_cache.de_tensor(bundle)
            de = de_tensor.reshape(de_tensor.shape[0], -1)
            metric_sets.extend(
                run_feature_classification(
                    de,
                    bundle.labels,
                    bundle.subjects,
                    bundle.trial_ids,
                    splits,
                    config,
                    result.run_dir,
                    "de",
                    classifier=classifier,
                )
            )
    if model in {"all", "deep"}:
        metric_sets.extend(
            run_torch_classification(
                bundle,
                splits,
                config,
                result.run_dir,
                deep_network=deep_network,
                protocol=protocol,
                input_kind=input_kind,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                device=device,
            )
        )
    _write_run_summary(
        result.run_dir,
        config,
        task,
        model,
        len(bundle.labels),
        len(splits),
        feature_kind,
        classifier,
        deep_network,
        protocol,
        input_kind,
        sequence_length,
        sequence_stride,
        device,
    )
    _append_metric_report(result.run_dir, task, config.evaluation.split_mode, len(bundle.labels), len(splits), metric_sets)


def _build_window_dataset(result: AuditResult, config: EEGConfig, task: str):
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
            raw_cache[trial.subject_key] = preprocess_bdf_to_raw(bdf_path, config)
        if not channel_names:
            channel_names = list(raw_cache[trial.subject_key].ch_names)
        trial_windows = list(extract_trial_windows(raw_cache[trial.subject_key], trial, config))
        for order, window in enumerate(trial_windows):
            windows.append(window)
            labels.append(label)
            subjects.append(trial.subject_key)
            trial_ids.append(trial.trial_id)
            window_order.append(order)
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to build EEG window datasets.") from exc
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


def _label_for_task(trial, task: str) -> str | None:
    if task == "category":
        return trial.category if trial.category in {"positive", "neutral", "negative"} else None
    if task == "valence_binary":
        return _rating_to_binary(trial.valence)
    if task == "arousal_binary":
        return _rating_to_binary(trial.arousal)
    raise ValueError(f"Unknown task: {task}")


def _rating_to_binary(value: int | None) -> str | None:
    if value is None:
        return None
    if value >= 4:
        return "high"
    if value <= 2:
        return "low"
    return None


def _make_splits(subjects, trial_ids, labels, config: EEGConfig):
    split_mode = config.evaluation.split_mode
    if split_mode == "loso":
        return make_loso_splits(subjects)
    if split_mode == "subject_dependent":
        return make_subject_dependent_splits(subjects, trial_ids)
    if split_mode == "window_kfold":
        return make_window_kfold_splits(labels, n_splits=10, random_seed=config.random_seed)
    raise ValueError(f"Unknown split mode: {split_mode}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_summary(
    run_dir: Path,
    config: EEGConfig,
    task: str,
    model: str,
    n_windows: int,
    n_splits: int,
    feature_kind: str,
    classifier: str,
    deep_network: str,
    protocol: str,
    input_kind: str,
    sequence_length: int,
    sequence_stride: int,
    device: str,
) -> None:
    payload = {
        "task": task,
        "model": model,
        "n_windows": n_windows,
        "n_splits": n_splits,
        "feature_kind": feature_kind,
        "classifier": classifier,
        "deep_network": deep_network,
        "protocol": protocol,
        "input_kind": input_kind,
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "device": device,
        "config": asdict(config),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, default=str, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_metric_report(
    run_dir: Path,
    task: str,
    split_mode: str,
    n_windows: int,
    n_splits: int,
    metrics: list[dict[str, object]],
) -> None:
    summary = _summarize_metrics(metrics)
    _write_csv(run_dir / "metrics_summary.csv", summary)
    lines = [
        "",
        "## Classification Results",
        "",
        f"- Task: {task}",
        f"- Split mode: {split_mode}",
        f"- Windows: {n_windows}",
        f"- Splits: {n_splits}",
        f"- Feature input normalization: per-subject per-channel z-score before PSD/DE",
        "",
        "| Feature | Model | Protocol | Folds | Balanced Acc Mean | Macro F1 Mean |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    if not summary:
        lines.append("| n/a | n/a | n/a | 0 | n/a | n/a |")
    for row in summary:
        lines.append(
            f"| {row['feature']} | {row['model']} | {row['protocol']} | {row['folds']} | "
            f"{float(row['balanced_accuracy_mean']):.4f} | {float(row['macro_f1_mean']):.4f} |"
        )
    if split_mode == "window_kfold":
        lines.extend(
            [
                "",
                "Note: window_kfold shuffles individual windows before 10-fold validation. "
                "Different windows from the same video trial may appear in both train and test folds.",
            ]
        )
    if any(item.get("validation_source") == "test_fold" for item in metrics):
        lines.extend(
            [
                "",
                "Note: deep-model early stopping used each fold's test set as the validation set.",
            ]
        )
    if any(item.get("uses_test_x_unlabeled") for item in metrics):
        lines.extend(
            [
                "",
                "Note: transductive domain-adaptation models used test-fold EEG features without labels during training.",
            ]
        )
    report_path = run_dir / "report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    prefix = existing.split("\n## Classification Results", 1)[0].rstrip()
    report_path.write_text(prefix + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _summarize_metrics(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for item in metrics:
        if "balanced_accuracy" not in item:
            continue
        key = (
            str(item.get("feature", "raw")),
            str(item.get("model", "unknown")),
            str(item.get("protocol", "supervised")),
        )
        grouped.setdefault(key, []).append(item)
    rows: list[dict[str, object]] = []
    for (feature, model, protocol), items in sorted(grouped.items()):
        balanced = [float(item["balanced_accuracy"]) for item in items]
        macro_f1 = [float(item["macro_f1"]) for item in items]
        rows.append(
            {
                "feature": feature,
                "model": model,
                "protocol": protocol,
                "folds": len(items),
                "balanced_accuracy_mean": sum(balanced) / len(balanced),
                "macro_f1_mean": sum(macro_f1) / len(macro_f1),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

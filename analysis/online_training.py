from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.Net import DEEP_MODEL_NAMES, get_model_spec, build_torch_model
from analysis.audit import run_audit
from analysis.config import EEGConfig, load_config
from analysis.eeg_pipeline import _build_window_dataset
from analysis.online_preprocessing import normalize_windows_zscore, probability_to_score
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

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "train":
        train_online_models(
            config,
            network=args.network,
            device=args.device,
            output_dir=Path(args.output_dir),
            run_id=args.run_id,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = run_audit(config, run_id=run_id or f"online_train_{format_run_timestamp()}")
    if not audit.valid_trials:
        raise RuntimeError(f"No valid aligned EEG trials. See {audit.run_dir / 'report.md'}")

    task_summaries: dict[str, object] = {}
    artifact_paths: dict[str, str] = {}
    for task_name, task_info in ONLINE_TASKS.items():
        bundle = _build_window_dataset(audit, config, task_info["pipeline_task"])
        if not bundle.labels:
            raise RuntimeError(f"No windows available after dropping neutral ratings for {task_name}.")

        x_np = normalize_windows_zscore(bundle.windows)
        y_np = np.asarray([1 if label == "high" else 0 for label in bundle.labels], dtype="float32")
        if len(set(y_np.tolist())) < 2:
            raise RuntimeError(f"{task_name} has fewer than two binary classes after filtering.")

        model, history = _train_single_binary_model(
            x_np,
            y_np,
            network=network,
            config=config,
            device=device_obj,
        )
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
                "score_formula": "1 + 4 * sigmoid(logit)",
                "low_label": task_info["low_label"],
                "high_label": task_info["high_label"],
            },
            artifact_path,
        )
        artifact_paths[task_name] = artifact_name
        positives = int(y_np.sum())
        negatives = int(len(y_np) - positives)
        task_summaries[task_name] = {
            "task": task_name,
            "pipeline_task": task_info["pipeline_task"],
            "network": network,
            "artifact": artifact_name,
            "n_windows": int(len(y_np)),
            "n_trials": int(len(set(bundle.trial_ids))),
            "n_subjects": int(len(set(bundle.subjects))),
            "class_counts": {
                task_info["low_label"]: negatives,
                task_info["high_label"]: positives,
            },
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
            "uses_all_available_binary_windows": True,
            "neutral_rating_3_dropped": True,
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


def _train_single_binary_model(x_np, y_np, *, network: str, config: EEGConfig, device):
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
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else None})
    return model, history


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


def _write_report(path: Path, metadata: dict[str, object]) -> None:
    lines = [
        "# Online Valence/Arousal Deployment Training",
        "",
        f"- Network: {metadata['network']}",
        f"- Normalization: {metadata['preprocessing']['normalization']}",
        f"- Device: {metadata['training']['device_resolved']}",
        "- Final artifacts are trained on all available non-neutral binary windows.",
        "- This deployment report is not a strict held-out generalization estimate.",
        "",
        "## Tasks",
        "",
    ]
    for task_name, summary in metadata["tasks"].items():
        counts = summary["class_counts"]
        lines.extend(
            [
                f"### {task_name}",
                "",
                f"- Artifact: {summary['artifact']}",
                f"- Windows: {summary['n_windows']}",
                f"- Trials: {summary['n_trials']}",
                f"- Subjects: {summary['n_subjects']}",
                f"- Class counts: {counts}",
                f"- Final loss: {summary['final_loss']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

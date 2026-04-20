from __future__ import annotations

import copy
import json
import warnings
from itertools import cycle
from pathlib import Path
from typing import Sequence

from analysis.Net import DEEP_MODEL_NAMES, build_torch_model, get_model_spec, list_models
from analysis.config import EEGConfig
from analysis.eeg_dataset import FeatureCache, WindowDatasetBundle, make_model_input
from analysis.splits import (
    EvaluationSplit,
    assert_no_split_leakage,
    make_loso_splits,
    make_subject_dependent_splits,
    make_window_kfold_splits,
)


def run_torch_classification(
    bundle: WindowDatasetBundle,
    splits: Sequence[EvaluationSplit],
    config: EEGConfig,
    output_dir: Path,
    *,
    deep_network: str = "shallow_convnet",
    protocol: str = "supervised",
    input_kind: str = "auto",
    sequence_length: int = 9,
    sequence_stride: int = 1,
    device: str = "auto",
) -> list[dict[str, object]]:
    try:
        import numpy as np
        import torch
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    except ModuleNotFoundError:
        output_dir.mkdir(parents=True, exist_ok=True)
        skipped = [{"model": deep_network, "status": "skipped_missing_torch_numpy_or_sklearn"}]
        (output_dir / "torch_metrics.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        return skipped

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.benchmark = True
    device_obj = _resolve_device(device, torch)
    feature_cache = FeatureCache(config, output_dir)
    model_names = _resolve_model_names(deep_network, protocol)
    label_names = sorted(set(bundle.labels))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    results: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for model_name in model_names:
        spec = get_model_spec(model_name)
        if spec.protocol != protocol:
            if deep_network == "all":
                skipped.append(
                    {
                        "model": model_name,
                        "status": "skipped_protocol_mismatch",
                        "required_protocol": spec.protocol,
                        "requested_protocol": protocol,
                    }
                )
                continue
            raise ValueError(f"{model_name} requires protocol={spec.protocol}, got {protocol}")
        selected_input_kind = spec.input_kind if input_kind == "auto" else input_kind
        if selected_input_kind != spec.input_kind:
            raise ValueError(f"{model_name} requires input_kind={spec.input_kind}, got {selected_input_kind}")
        model_input = make_model_input(
            bundle,
            input_kind=selected_input_kind,
            feature_cache=feature_cache,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
        )
        if len(model_input.labels) == 0:
            skipped.append({"model": model_name, "status": "skipped_no_samples_after_input_build"})
            continue
        effective_splits = _remake_splits(model_input.subjects, model_input.trial_ids, model_input.labels, config)
        if not effective_splits:
            skipped.append({"model": model_name, "status": "skipped_no_valid_splits"})
            continue
        x = torch.tensor(np.asarray(model_input.x), dtype=torch.float32)
        y = torch.tensor([label_to_index[label] for label in model_input.labels], dtype=torch.long)
        n_channels = int(x.shape[-2]) if selected_input_kind == "de_sequence" else int(x.shape[1])
        n_samples = int(x.shape[-1]) if selected_input_kind == "raw" else None
        n_bands = int(x.shape[-1]) if selected_input_kind in {"de", "de_sequence"} else len(bundle.band_names)
        for split in effective_splits:
            try:
                assert_no_split_leakage(split, model_input.subjects, model_input.trial_ids, config.evaluation.split_mode)
                metric = _train_one_split(
                    model_name,
                    protocol,
                    x,
                    y,
                    model_input.labels,
                    model_input.subjects,
                    model_input.trial_ids,
                    split,
                    config,
                    device_obj,
                    n_channels=n_channels,
                    n_classes=len(label_names),
                    n_samples=n_samples,
                    n_bands=n_bands,
                    sequence_length=sequence_length,
                    label_names=label_names,
                    label_to_index=label_to_index,
                    balanced_accuracy_score=balanced_accuracy_score,
                    f1_score=f1_score,
                    confusion_matrix=confusion_matrix,
                )
                metric.update(
                    {
                        "feature": selected_input_kind,
                        "model": model_name,
                        "protocol": protocol,
                        "input_kind": selected_input_kind,
                        "requires_target_unlabeled": spec.requires_target_unlabeled,
                        "source": spec.source,
                    }
                )
                results.append(metric)
            except ValueError as exc:
                skipped.append({"model": model_name, "split": split.name, "status": "skipped", "reason": str(exc)})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "torch_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if skipped:
        (output_dir / "skipped_models.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(
        json.dumps(
            {
                "deep_network": deep_network,
                "protocol": protocol,
                "input_kind": input_kind,
                "sequence_length": sequence_length,
                "sequence_stride": sequence_stride,
                "requested_device": device,
                "resolved_device": str(device_obj),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else None,
                "available_models": [spec.__dict__ for spec in list_models()],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return results + skipped


def _train_one_split(
    model_name: str,
    protocol: str,
    x,
    y,
    labels: Sequence[str],
    subjects: Sequence[str],
    trial_ids: Sequence[str],
    split: EvaluationSplit,
    config: EEGConfig,
    device,
    *,
    n_channels: int,
    n_classes: int,
    n_samples: int | None,
    n_bands: int,
    sequence_length: int,
    label_names: list[str],
    label_to_index: dict[str, int],
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    train_idx = list(split.train_indices)
    test_idx = list(split.test_indices)
    if len({labels[index] for index in train_idx}) < 2:
        raise ValueError("training fold has fewer than two classes")
    if protocol == "transductive_da" and config.evaluation.split_mode != "loso":
        raise ValueError("transductive_da is only supported for split_mode=loso")
    num_domains = max(2, len({subjects[index] for index in train_idx}))
    model = build_torch_model(
        model_name,
        n_channels=n_channels,
        n_classes=n_classes,
        n_samples=n_samples,
        n_bands=n_bands,
        sequence_length=sequence_length,
        num_domains=num_domains,
    ).to(device)
    if protocol == "supervised":
        validation_source = "test_fold"
        _train_supervised(model, x, y, train_idx, test_idx, config, device)
        uses_test_x_unlabeled = False
        uses_test_y_for_training = True
    elif protocol == "source_dg":
        validation_source = "test_fold"
        _train_source_dg(model, x, y, subjects, train_idx, test_idx, config, device)
        uses_test_x_unlabeled = False
        uses_test_y_for_training = True
    elif protocol == "transductive_da":
        validation_source = "source_loss"
        _train_transductive_da(model, x, y, train_idx, test_idx, config, device)
        uses_test_x_unlabeled = True
        uses_test_y_for_training = False
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    pred, truth = _predict(model, x, y, test_idx, device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", message="A single label was found")
        balanced_accuracy = balanced_accuracy_score(truth, pred)
        macro_f1 = f1_score(truth, pred, average="macro", zero_division=0)
        matrix = confusion_matrix(truth, pred, labels=list(range(len(label_names)))).tolist()
    return {
        "split": split.name,
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "n_train": len(train_idx),
        "n_val": len(test_idx) if validation_source == "test_fold" else 0,
        "n_test": len(test_idx),
        "validation_source": validation_source,
        "uses_test_x_unlabeled": uses_test_x_unlabeled,
        "uses_test_y_for_training": uses_test_y_for_training,
        "labels": label_names,
        "test_classes": sorted({labels[index] for index in test_idx}),
        "confusion_matrix": matrix,
    }


def _train_supervised(model, x, y, train_idx, val_idx, config: EEGConfig, device) -> None:
    import torch
    import torch.nn.functional as F

    train_loader = _loader(x, y, train_idx, config.models.deep_batch_size, shuffle=True, device=device)
    if hasattr(model, "reconstruction_loss"):
        _pretrain_reconstruction(model, train_loader, config, device)
    optimizer = _optimizer(model, config)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _ in range(config.models.deep_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = _to_device(batch_x, device), _to_device(batch_y, device)
            optimizer.zero_grad()
            loss = F.cross_entropy(_logits(model(batch_x)), batch_y)
            loss.backward()
            optimizer.step()
        val_loss = _classification_loss(model, x, y, val_idx, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.models.deep_patience:
                break
    if best_state:
        model.load_state_dict(best_state)


def _train_source_dg(model, x, y, subjects, train_idx, val_idx, config: EEGConfig, device) -> None:
    import torch
    import torch.nn.functional as F

    domains = {subject: index for index, subject in enumerate(sorted({subjects[index] for index in train_idx}))}
    domain_y = torch.tensor([domains.get(subjects[index], 0) for index in range(len(subjects))], dtype=torch.long)
    train_loader = _loader(x, y, train_idx, config.models.deep_batch_size, shuffle=True, aux=domain_y, device=device)
    optimizer = _optimizer(model, config)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(config.models.deep_epochs):
        alpha = _domain_alpha(epoch, config.models.deep_epochs)
        model.train()
        for batch_x, batch_y, batch_domain in train_loader:
            batch_x, batch_y, batch_domain = _to_device(batch_x, device), _to_device(batch_y, device), _to_device(batch_domain, device)
            optimizer.zero_grad()
            output = model(batch_x, alpha=alpha)
            loss = F.cross_entropy(output["logits"], batch_y) + 0.2 * F.cross_entropy(
                output["domain_logits"], batch_domain
            )
            loss.backward()
            optimizer.step()
        val_loss = _classification_loss(model, x, y, val_idx, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.models.deep_patience:
                break
    if best_state:
        model.load_state_dict(best_state)


def _train_transductive_da(model, x, y, train_idx, target_idx, config: EEGConfig, device) -> None:
    import torch
    import torch.nn.functional as F

    source_loader = _loader(x, y, train_idx, config.models.deep_batch_size, shuffle=True, device=device)
    target_loader = _loader(x, y, target_idx, config.models.deep_batch_size, shuffle=True, device=device)
    if len(source_loader) == 0 or len(target_loader) == 0:
        return
    optimizer = _optimizer(model, config)
    best_state = None
    best_loss = float("inf")
    for epoch in range(config.models.deep_epochs):
        alpha = _domain_alpha(epoch, config.models.deep_epochs)
        model.train()
        total_loss = 0.0
        batches = 0
        for (source_x, source_y), (target_x, _) in zip(source_loader, cycle(target_loader)):
            source_x, source_y = _to_device(source_x, device), _to_device(source_y, device)
            target_x = _to_device(target_x, device)
            optimizer.zero_grad()
            output = model(source_x, target_x, alpha=alpha)
            domain_y = torch.cat(
                [
                    torch.zeros(len(source_x), dtype=torch.long),
                    torch.ones(len(target_x), dtype=torch.long),
                ]
            ).to(device)
            loss = F.cross_entropy(output["logits"], source_y) + 0.2 * F.cross_entropy(
                output["domain_logits"], domain_y
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            if batches >= len(source_loader):
                break
        source_loss = total_loss / max(batches, 1)
        if source_loss < best_loss:
            best_loss = source_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)


def _pretrain_reconstruction(model, train_loader, config: EEGConfig, device) -> None:
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=config.models.deep_learning_rate)
    epochs = max(1, min(3, config.models.deep_epochs // 10 or 1))
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            batch_x = _to_device(batch[0], device)
            optimizer.zero_grad()
            loss = model.reconstruction_loss(batch_x)
            loss.backward()
            optimizer.step()


def _classification_loss(model, x, y, indices, device) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        batch_x = _to_device(x[list(indices)], device)
        batch_y = _to_device(y[list(indices)], device)
        return float(F.cross_entropy(_predict_logits(model, batch_x), batch_y).item())


def _predict(model, x, y, indices, device):
    import torch

    model.eval()
    with torch.no_grad():
        logits = _predict_logits(model, _to_device(x[list(indices)], device))
    return logits.argmax(dim=1).cpu().numpy(), y[list(indices)].cpu().numpy()


def _predict_logits(model, batch_x):
    if hasattr(model, "predict_logits"):
        return model.predict_logits(batch_x)
    return _logits(model(batch_x))


def _logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output


def _loader(x, y, indices, batch_size: int, shuffle: bool, aux=None, drop_last: bool = False, device=None):
    import torch

    tensors = [x[list(indices)], y[list(indices)]]
    if aux is not None:
        tensors.append(aux[list(indices)])
    dataset = torch.utils.data.TensorDataset(*tensors)
    pin_memory = bool(getattr(device, "type", None) == "cuda")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def _optimizer(model, config: EEGConfig):
    import torch

    return torch.optim.AdamW(model.parameters(), lr=config.models.deep_learning_rate, weight_decay=1e-4)


def _domain_alpha(epoch: int, epochs: int) -> float:
    import math

    p = epoch / max(epochs - 1, 1)
    return float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


def _resolve_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but the current PyTorch build cannot use CUDA. "
            "Install a CUDA-enabled torch build, or set models.deep_device/cmd --device to cpu."
        )
    return resolved


def _to_device(tensor, device):
    return tensor.to(device, non_blocking=bool(getattr(device, "type", None) == "cuda"))


def _resolve_model_names(deep_network: str, protocol: str) -> list[str]:
    if deep_network == "all":
        return [spec.name for spec in list_models(protocol=protocol)]
    if deep_network not in DEEP_MODEL_NAMES:
        raise ValueError(f"Unknown deep network: {deep_network}. Available: {', '.join(DEEP_MODEL_NAMES)}")
    return [deep_network]


def _remake_splits(subjects, trial_ids, labels, config: EEGConfig):
    split_mode = config.evaluation.split_mode
    if split_mode == "loso":
        return make_loso_splits(subjects)
    if split_mode == "subject_dependent":
        return make_subject_dependent_splits(subjects, trial_ids)
    if split_mode == "window_kfold":
        return make_window_kfold_splits(labels, n_splits=10, random_seed=config.random_seed)
    raise ValueError(f"Unknown split mode: {split_mode}")

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

from analysis.Net import DEEP_MODEL_NAMES, build_deep_model
from analysis.config import EEGConfig
from analysis.splits import EvaluationSplit, assert_no_split_leakage


def run_deep_classification(
    windows,
    y: Sequence[str],
    subjects: Sequence[str],
    trial_ids: Sequence[str],
    splits: Sequence[EvaluationSplit],
    config: EEGConfig,
    output_dir: Path,
    deep_network: str | None = None,
) -> list[dict[str, object]]:
    try:
        import numpy as np
        import torch
        from sklearn.metrics import balanced_accuracy_score, f1_score
    except ModuleNotFoundError:
        output_dir.mkdir(parents=True, exist_ok=True)
        skipped = [{"model": "shallow_convnet", "status": "skipped_missing_torch_or_sklearn"}]
        (output_dir / "deep_metrics.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        return skipped

    torch.manual_seed(config.random_seed)
    X = torch.tensor(np.asarray(windows), dtype=torch.float32)
    label_names = sorted(set(y))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    y_index = torch.tensor([label_to_index[label] for label in y], dtype=torch.long)
    results: list[dict[str, object]] = []
    model_names = _resolve_deep_model_names(deep_network, config.models.deep_models)
    for split in splits:
        assert_no_split_leakage(split, subjects, trial_ids, config.evaluation.split_mode)
        train_idx = list(split.train_indices)
        test_idx = list(split.test_indices)
        if len({y[index] for index in train_idx}) < 2:
            continue
        for model_name in model_names:
            model = build_deep_model(model_name, n_channels=X.shape[1], n_classes=len(label_names), n_samples=X.shape[2])
            val_idx = test_idx
            _train_torch_model(model, X[train_idx], y_index[train_idx], X[val_idx], y_index[val_idx], config)
            with torch.no_grad():
                pred = model(X[test_idx]).argmax(dim=1).cpu().numpy()
            truth = y_index[test_idx].cpu().numpy()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                balanced_accuracy = balanced_accuracy_score(truth, pred)
                macro_f1 = f1_score(truth, pred, average="macro", zero_division=0)
            results.append(
                {
                    "model": model_name,
                    "split": split.name,
                    "balanced_accuracy": float(balanced_accuracy),
                    "macro_f1": float(macro_f1),
                    "n_train": len(train_idx),
                    "n_train_core": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "validation_source": "test_fold",
                    "labels": label_names,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deep_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _resolve_deep_model_names(deep_network: str | None, configured: Sequence[str]) -> list[str]:
    if deep_network and deep_network != "all":
        if deep_network not in DEEP_MODEL_NAMES:
            raise ValueError(f"Unknown deep network: {deep_network}. Available: {', '.join(DEEP_MODEL_NAMES)}")
        return [deep_network]
    return [name for name in configured if name in DEEP_MODEL_NAMES]


def _train_torch_model(model, X_train, y_train, X_val, y_val, config: EEGConfig) -> None:
    import torch

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.models.deep_learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.models.deep_batch_size, shuffle=True)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for _ in range(config.models.deep_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), y_val).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.models.deep_patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

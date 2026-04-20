from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvaluationSplit:
    name: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def make_loso_splits(subjects: Sequence[str]) -> list[EvaluationSplit]:
    unique_subjects = sorted(set(subjects))
    splits: list[EvaluationSplit] = []
    for subject in unique_subjects:
        test = tuple(index for index, value in enumerate(subjects) if value == subject)
        train = tuple(index for index, value in enumerate(subjects) if value != subject)
        if train and test:
            splits.append(EvaluationSplit(name=f"loso_{subject}", train_indices=train, test_indices=test))
    return splits


def make_subject_dependent_splits(subjects: Sequence[str], trial_ids: Sequence[str]) -> list[EvaluationSplit]:
    splits: list[EvaluationSplit] = []
    for subject in sorted(set(subjects)):
        subject_indices = [index for index, value in enumerate(subjects) if value == subject]
        subject_trials = sorted({trial_ids[index] for index in subject_indices})
        if len(subject_trials) < 2:
            continue
        for trial in subject_trials:
            test = tuple(index for index in subject_indices if trial_ids[index] == trial)
            train = tuple(index for index in subject_indices if trial_ids[index] != trial)
            if train and test:
                splits.append(EvaluationSplit(name=f"{subject}_leave_{trial}", train_indices=train, test_indices=test))
    return splits


def make_window_kfold_splits(
    labels: Sequence[str],
    n_splits: int = 10,
    random_seed: int = 42,
) -> list[EvaluationSplit]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    n_samples = len(labels)
    if n_samples < 2:
        return []
    n_splits = min(n_splits, n_samples)
    rng = random.Random(random_seed)
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(index)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for indices in buckets.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            folds[offset % n_splits].append(index)
    all_indices = set(range(n_samples))
    splits: list[EvaluationSplit] = []
    for fold_index, test_indices in enumerate(folds, start=1):
        if not test_indices:
            continue
        test = tuple(sorted(test_indices))
        train = tuple(sorted(all_indices - set(test)))
        if train:
            splits.append(EvaluationSplit(name=f"window_kfold_{fold_index:02d}", train_indices=train, test_indices=test))
    return splits


def assert_no_split_leakage(split: EvaluationSplit, subjects: Sequence[str], trial_ids: Sequence[str], mode: str) -> None:
    train_index_set = set(split.train_indices)
    test_index_set = set(split.test_indices)
    index_overlap = train_index_set & test_index_set
    if index_overlap:
        raise ValueError(f"Index leakage in split {split.name}: {sorted(index_overlap)}")
    if mode == "window_kfold":
        return
    train_trials = {trial_ids[index] for index in split.train_indices}
    test_trials = {trial_ids[index] for index in split.test_indices}
    overlap = train_trials & test_trials
    if overlap:
        raise ValueError(f"Trial leakage in split {split.name}: {sorted(overlap)}")
    if mode == "loso":
        train_subjects = {subjects[index] for index in split.train_indices}
        test_subjects = {subjects[index] for index in split.test_indices}
        subject_overlap = train_subjects & test_subjects
        if subject_overlap:
            raise ValueError(f"Subject leakage in split {split.name}: {sorted(subject_overlap)}")

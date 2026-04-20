from analysis.splits import assert_no_split_leakage, make_loso_splits, make_subject_dependent_splits, make_window_kfold_splits


def test_loso_splits_never_share_subjects() -> None:
    subjects = ["sub1", "sub1", "sub2", "sub2"]
    trials = ["t1", "t1", "t2", "t2"]

    splits = make_loso_splits(subjects)

    assert len(splits) == 2
    for split in splits:
        assert_no_split_leakage(split, subjects, trials, "loso")


def test_subject_dependent_splits_leave_whole_trial_out() -> None:
    subjects = ["sub1", "sub1", "sub1", "sub1"]
    trials = ["t1", "t1", "t2", "t2"]

    splits = make_subject_dependent_splits(subjects, trials)

    assert len(splits) == 2
    for split in splits:
        assert_no_split_leakage(split, subjects, trials, "subject_dependent")


def test_window_kfold_splits_shuffle_windows_and_allow_trial_overlap() -> None:
    labels = ["a"] * 10 + ["b"] * 10
    subjects = ["sub3"] * 20
    trials = ["t1"] * 10 + ["t2"] * 10

    splits = make_window_kfold_splits(labels, n_splits=10, random_seed=42)

    assert len(splits) == 10
    assert all(len(split.test_indices) == 2 for split in splits)
    for split in splits:
        assert_no_split_leakage(split, subjects, trials, "window_kfold")

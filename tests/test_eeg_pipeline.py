from analysis.eeg_pipeline import parse_subject_key_filters


def test_parse_subject_key_filters_accepts_single_and_multiple_forms() -> None:
    assert parse_subject_key_filters("003", ["sub4,5", "sub5"]) == ("sub3", "sub4", "sub5")


def test_parse_subject_key_filters_handles_empty_values() -> None:
    assert parse_subject_key_filters(None, ["sub3,,", " "]) == ("sub3",)

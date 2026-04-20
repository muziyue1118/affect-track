from datetime import datetime
from pathlib import Path

from analysis.audit import build_trial_manifest
from analysis.bdf import BDFHeader
from analysis.config import EEGConfig, SegmentationConfig
from analysis.labels import LabelRecord


def test_trial_manifest_rejects_out_of_bounds_label() -> None:
    header = _header(duration_s=120)
    label = _label(start_time="E202604101700000000", end_time="E202604101705000000")
    config = EEGConfig(segmentation=SegmentationConfig(trim_start_s=30, trim_end_s=10, duration_s="full", min_trial_s=60))

    trials = build_trial_manifest([header], [label], config)

    assert trials[0].status == "invalid"
    assert trials[0].reason == "trial_outside_eeg_recording"


def test_trial_manifest_accepts_valid_label_and_trims_start_and_end() -> None:
    header = _header(duration_s=600)
    label = _label(start_time="E202604101608000000", end_time="E202604101612000000")
    config = EEGConfig(segmentation=SegmentationConfig(trim_start_s=30, trim_end_s=10, duration_s="full", min_trial_s=60))

    trial = build_trial_manifest([header], [label], config)[0]

    assert trial.status == "valid"
    assert trial.raw_start_s == 20
    assert trial.effective_start_s == 50
    assert trial.effective_end_s == 250
    assert trial.effective_duration_s == 200


def test_trial_manifest_can_still_cap_duration_for_exploratory_runs() -> None:
    header = _header(duration_s=600)
    label = _label(start_time="E202604101608000000", end_time="E202604101612000000")
    config = EEGConfig(segmentation=SegmentationConfig(trim_start_s=30, trim_end_s=10, duration_s=180, min_trial_s=60))

    trial = build_trial_manifest([header], [label], config)[0]

    assert trial.status == "valid"
    assert trial.effective_start_s == 50
    assert trial.effective_end_s == 230
    assert trial.effective_duration_s == 180


def _header(duration_s: float) -> BDFHeader:
    return BDFHeader(
        path=Path("data.bdf"),
        subject_id="sub1",
        subject_key="sub1",
        start_time=datetime(2026, 4, 10, 16, 7, 40),
        header_bytes=8448,
        n_records=int(duration_s),
        record_duration_s=1.0,
        n_signals=32,
        labels=tuple(f"Ch{i}" for i in range(32)),
        samples_per_record=tuple(1000 for _ in range(32)),
        duration_s=duration_s,
    )


def _label(start_time: str, end_time: str) -> LabelRecord:
    return LabelRecord(
        row_number=2,
        subject_id="001",
        subject_key="sub1",
        video_name="positive_1.mp4",
        category="positive",
        start_time=start_time,
        end_time=end_time,
        valence=4,
        arousal=2,
        saved_at=None,
        schema_status="current",
    )

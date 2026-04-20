from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from analysis.bdf import BDFHeader, discover_bdf_headers
from analysis.config import EEGConfig
from analysis.labels import LabelRecord, load_label_records
from analysis.time_utils import format_run_timestamp


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    row_number: int
    subject_id: str
    subject_key: str
    video_name: str
    category: str
    start_time: str
    end_time: str | None
    valence: int | None
    arousal: int | None
    eeg_start_time: str | None
    raw_start_s: float | None
    raw_end_s: float | None
    effective_start_s: float | None
    effective_end_s: float | None
    effective_duration_s: float | None
    status: str
    reason: str

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"


@dataclass(frozen=True)
class AuditResult:
    run_dir: Path
    subject_rows: list[dict[str, object]]
    label_records: list[LabelRecord]
    trial_records: list[TrialRecord]

    @property
    def valid_trials(self) -> list[TrialRecord]:
        return [record for record in self.trial_records if record.is_valid]


def run_audit(config: EEGConfig, run_id: str | None = None) -> AuditResult:
    run_dir = config.output_dir / (run_id or format_run_timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    headers = discover_bdf_headers(config.data_dir)
    labels = load_label_records(config.labels_csv)
    subject_rows = _build_subject_rows(headers, labels)
    trials = build_trial_manifest(headers, labels, config)
    _write_dict_rows(run_dir / "data_audit.csv", subject_rows)
    _write_dict_rows(run_dir / "trial_manifest.csv", [asdict(trial) for trial in trials])
    _write_report(run_dir / "report.md", subject_rows, labels, trials, config)
    return AuditResult(run_dir=run_dir, subject_rows=subject_rows, label_records=labels, trial_records=trials)


def build_trial_manifest(headers: list[BDFHeader], labels: list[LabelRecord], config: EEGConfig) -> list[TrialRecord]:
    subjects = {header.subject_key: header for header in headers}
    trials: list[TrialRecord] = []
    for label in labels:
        trial_id = f"{label.subject_key}_{label.video_name}_{label.row_number}"
        header = subjects.get(label.subject_key)
        reason = ""
        status = "invalid"
        raw_start_s = raw_end_s = effective_start_s = effective_end_s = effective_duration_s = None
        eeg_start_time = header.start_time.isoformat(timespec="seconds") if header else None

        if header is None:
            reason = "no_matching_eeg_subject"
        elif not label.has_required_training_fields:
            reason = label.schema_status
        else:
            try:
                start_dt = label.start_dt
                end_dt = label.end_dt
            except ValueError as exc:
                reason = str(exc)
            else:
                if end_dt is None:
                    reason = "missing_end_time"
                else:
                    raw_start_s = (start_dt - header.start_time).total_seconds()
                    raw_end_s = (end_dt - header.start_time).total_seconds()
                    status, reason, effective_start_s, effective_end_s, effective_duration_s = _validate_trial_offsets(
                        raw_start_s,
                        raw_end_s,
                        header.duration_s,
                        config,
                    )

        trials.append(
            TrialRecord(
                trial_id=trial_id,
                row_number=label.row_number,
                subject_id=label.subject_id,
                subject_key=label.subject_key,
                video_name=label.video_name,
                category=label.category,
                start_time=label.start_time,
                end_time=label.end_time,
                valence=label.valence,
                arousal=label.arousal,
                eeg_start_time=eeg_start_time,
                raw_start_s=raw_start_s,
                raw_end_s=raw_end_s,
                effective_start_s=effective_start_s,
                effective_end_s=effective_end_s,
                effective_duration_s=effective_duration_s,
                status=status,
                reason=reason,
            )
        )
    return trials


def _validate_trial_offsets(
    raw_start_s: float,
    raw_end_s: float,
    eeg_duration_s: float,
    config: EEGConfig,
) -> tuple[str, str, float | None, float | None, float | None]:
    if raw_end_s <= raw_start_s:
        return "invalid", "non_positive_trial_duration", None, None, None
    if raw_start_s < 0 or raw_end_s > eeg_duration_s:
        return "invalid", "trial_outside_eeg_recording", None, None, None
    effective_start = raw_start_s + config.segmentation.trim_start_s
    trimmed_trial_end = raw_end_s - config.segmentation.trim_end_s
    if config.segmentation.duration_s == "full":
        effective_end = trimmed_trial_end
    else:
        effective_end = min(trimmed_trial_end, effective_start + float(config.segmentation.duration_s))
    effective_duration = effective_end - effective_start
    if effective_duration < config.segmentation.min_trial_s:
        return "invalid", "too_short_after_trim", effective_start, effective_end, effective_duration
    return "valid", "", effective_start, effective_end, effective_duration


def _build_subject_rows(headers: list[BDFHeader], labels: list[LabelRecord]) -> list[dict[str, object]]:
    label_counts: dict[str, int] = {}
    current_label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label.subject_key] = label_counts.get(label.subject_key, 0) + 1
        if label.has_required_training_fields:
            current_label_counts[label.subject_key] = current_label_counts.get(label.subject_key, 0) + 1
    rows: list[dict[str, object]] = []
    for header in headers:
        rows.append(
            {
                "subject_id": header.subject_id,
                "subject_key": header.subject_key,
                "bdf_path": str(header.path),
                "eeg_start_time": header.start_time.isoformat(timespec="seconds"),
                "duration_s": round(header.duration_s, 3),
                "n_channels": header.n_signals,
                "sample_rate_hz": round(header.sample_rate_hz, 3),
                "matched_label_count": label_counts.get(header.subject_key, 0),
                "trainable_label_count": current_label_counts.get(header.subject_key, 0),
                "status": "ok" if current_label_counts.get(header.subject_key, 0) else "no_trainable_labels",
            }
        )
    return rows


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    subject_rows: list[dict[str, object]],
    labels: list[LabelRecord],
    trials: list[TrialRecord],
    config: EEGConfig,
) -> None:
    valid = [trial for trial in trials if trial.is_valid]
    invalid = [trial for trial in trials if not trial.is_valid]
    reason_counts: dict[str, int] = {}
    for trial in invalid:
        reason_counts[trial.reason] = reason_counts.get(trial.reason, 0) + 1
    lines = [
        "# EEG Data Audit",
        "",
        f"- EEG subjects: {len(subject_rows)}",
        f"- Label rows: {len(labels)}",
        f"- Valid trials: {len(valid)}",
        f"- Invalid trials: {len(invalid)}",
        (
            f"- Default segment: trim_start {config.segmentation.trim_start_s}s, "
            f"trim_end {config.segmentation.trim_end_s}s, duration {config.segmentation.duration_s}"
        ),
        "",
        "## Invalid Trial Reasons",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(reason_counts.items())) if reason_counts else lines.append("- none")
    if not valid:
        lines.extend(
            [
                "",
                "## Training Status",
                "",
                "Training is refused because no valid trial can be aligned without risking label errors or data leakage.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

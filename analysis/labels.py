from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from analysis.time_utils import is_e_timestamp, parse_e_timestamp


@dataclass(frozen=True)
class LabelRecord:
    row_number: int
    subject_id: str
    subject_key: str
    video_name: str
    category: str
    start_time: str
    end_time: str | None
    valence: int | None
    arousal: int | None
    saved_at: str | None
    schema_status: str

    @property
    def start_dt(self) -> datetime:
        return parse_e_timestamp(self.start_time)

    @property
    def end_dt(self) -> datetime | None:
        return parse_e_timestamp(self.end_time) if self.end_time else None

    @property
    def has_required_training_fields(self) -> bool:
        return (
            self.schema_status in {"current", "header_missing_end_time_but_value_present"}
            and self.end_time is not None
            and self.valence is not None
            and self.arousal is not None
        )


def normalize_subject_id(value: str) -> str:
    text = str(value).strip().lower()
    match = re.search(r"(?:subject|subj|sub)?[_-]?0*(\d+)$", text)
    if match:
        return f"sub{int(match.group(1))}"
    return re.sub(r"[^a-z0-9]+", "", text)


def load_label_records(path: str | Path) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
        normalized_header = [name.strip() for name in header]
        for row_number, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            records.append(_parse_row(normalized_header, row, row_number))
    return records


def _parse_row(header: list[str], row: list[str], row_number: int) -> LabelRecord:
    index = {name: pos for pos, name in enumerate(header)}

    def get(name: str, default: str | None = None) -> str | None:
        pos = index.get(name)
        if pos is None or pos >= len(row):
            return default
        return row[pos].strip()

    subject_id = get("subject_id", "") or ""
    video_name = get("video_name", "") or ""
    category = get("category", "") or ""
    start_time = get("start_time", "") or ""
    saved_at = get("saved_at")
    end_time = get("end_time")
    valence_raw = get("valence")
    arousal_raw = get("arousal")
    schema_status = "current" if "end_time" in index else "old_missing_end_time"

    if "end_time" not in index and len(row) >= 8 and is_e_timestamp(row[4].strip()):
        end_time = row[4].strip()
        valence_raw = row[5].strip()
        arousal_raw = row[6].strip()
        saved_at = row[7].strip()
        schema_status = "header_missing_end_time_but_value_present"

    valence = _safe_int(valence_raw)
    arousal = _safe_int(arousal_raw)
    if not is_e_timestamp(start_time):
        schema_status = f"{schema_status};invalid_start_time"
    if end_time is not None and not is_e_timestamp(end_time):
        schema_status = f"{schema_status};invalid_end_time"

    return LabelRecord(
        row_number=row_number,
        subject_id=subject_id,
        subject_key=normalize_subject_id(subject_id),
        video_name=video_name,
        category=category,
        start_time=start_time,
        end_time=end_time,
        valence=valence,
        arousal=arousal,
        saved_at=saved_at,
        schema_status=schema_status,
    )


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None

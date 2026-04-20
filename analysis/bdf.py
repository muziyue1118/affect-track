from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from analysis.labels import normalize_subject_id
from analysis.time_utils import parse_e_timestamp


@dataclass(frozen=True)
class BDFHeader:
    path: Path
    subject_id: str
    subject_key: str
    start_time: datetime
    header_bytes: int
    n_records: int
    record_duration_s: float
    n_signals: int
    labels: tuple[str, ...]
    samples_per_record: tuple[int, ...]
    duration_s: float

    @property
    def sample_rate_hz(self) -> float:
        if not self.samples_per_record or self.record_duration_s <= 0:
            return 0.0
        return self.samples_per_record[0] / self.record_duration_s


def read_bdf_header(path: str | Path, info_path: str | Path | None = None) -> BDFHeader:
    bdf_path = Path(path)
    with bdf_path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) < 256:
            raise ValueError(f"BDF header too short: {bdf_path}")
        n_signals = _parse_int(fixed[252:256], default=0)
        header_bytes = _parse_int(fixed[184:192], default=256 + 256 * n_signals)
        handle.seek(0)
        header = handle.read(header_bytes)

    subject_id = _read_subject_from_info(info_path) or _parse_subject_from_folder(bdf_path.parent)
    start_time = _read_start_time_from_header(fixed) or _read_start_from_info(info_path) or _parse_start_from_folder(bdf_path.parent)
    n_records = _parse_int(fixed[236:244], default=0)
    record_duration_s = _parse_float(fixed[244:252], default=1.0)
    labels, samples_per_record = _read_signal_fields(header, n_signals)
    duration_s = max(0.0, n_records * record_duration_s)
    return BDFHeader(
        path=bdf_path,
        subject_id=subject_id,
        subject_key=normalize_subject_id(subject_id),
        start_time=start_time,
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration_s=record_duration_s,
        n_signals=n_signals,
        labels=tuple(labels),
        samples_per_record=tuple(samples_per_record),
        duration_s=duration_s,
    )


def discover_bdf_headers(data_dir: str | Path) -> list[BDFHeader]:
    headers: list[BDFHeader] = []
    for folder in sorted(Path(data_dir).glob("sub*_E*")):
        bdf_path = folder / "data.bdf"
        if bdf_path.exists():
            headers.append(read_bdf_header(bdf_path, folder / "recordInformation.json"))
    return headers


def _read_signal_fields(header: bytes, n_signals: int) -> tuple[list[str], list[int]]:
    offset = 256
    labels: list[str] = []
    samples: list[int] = []
    for name, width in [
        ("labels", 16),
        ("transducer", 80),
        ("physical_dimension", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefiltering", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    ]:
        values = [
            header[offset + i * width : offset + (i + 1) * width].decode("ascii", errors="ignore").strip()
            for i in range(n_signals)
        ]
        if name == "labels":
            labels = values
        if name == "samples_per_record":
            samples = [_parse_int(value.encode("ascii", errors="ignore"), default=0) for value in values]
        offset += n_signals * width
    return labels, samples


def _read_start_time_from_header(header: bytes) -> datetime | None:
    date_text = header[168:176].decode("ascii", errors="ignore").strip()
    time_text = header[176:184].decode("ascii", errors="ignore").strip()
    try:
        day, month, year = [int(part) for part in date_text.split(".")]
        hour, minute, second = [int(part) for part in time_text.split(".")]
    except ValueError:
        return None
    year += 2000 if year < 80 else 1900
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def _read_subject_from_info(info_path: str | Path | None) -> str | None:
    info = _read_json(info_path)
    if not info:
        return None
    return str(info.get("PatientID") or "").strip() or None


def _read_start_from_info(info_path: str | Path | None) -> datetime | None:
    info = _read_json(info_path)
    if not info:
        return None
    exam_time = str(info.get("ExamTime") or "").strip()
    if exam_time:
        try:
            return datetime.fromisoformat(exam_time)
        except ValueError:
            pass
    exam_id = str(info.get("ExamID") or "").strip()
    if exam_id:
        try:
            return parse_e_timestamp(exam_id)
        except ValueError:
            return None
    return None


def _read_json(info_path: str | Path | None) -> dict | None:
    if info_path is None:
        return None
    path = Path(info_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _parse_subject_from_folder(folder: Path) -> str:
    return folder.name.split("_", 1)[0]


def _parse_start_from_folder(folder: Path) -> datetime:
    marker = folder.name.rsplit("_", 1)[-1]
    return parse_e_timestamp(marker)


def _parse_int(raw: bytes, default: int) -> int:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _parse_float(raw: bytes, default: float) -> float:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default

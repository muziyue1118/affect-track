import shutil
import uuid
from pathlib import Path

from analysis.labels import load_label_records, normalize_subject_id


def test_normalize_subject_id_variants() -> None:
    assert normalize_subject_id("sub1") == "sub1"
    assert normalize_subject_id("001") == "sub1"
    assert normalize_subject_id("sub_004") == "sub4"


def test_load_label_records_handles_old_header_with_new_row() -> None:
    root = _runtime_root("labels")
    try:
        csv_path = root / "offline_records.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "subject_id,video_name,category,start_time,valence,arousal,saved_at",
                    "test_sub,neutral_1.mp4,neutral,E202604092137025582,5,5,2026-04-09T21:43:41.762",
                    "001,positive_1.mp4,positive,E202604101442306475,E202604101450102711,4,2,2026-04-10T14:51:14.586",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        records = load_label_records(csv_path)

        assert records[0].end_time is None
        assert records[0].schema_status == "old_missing_end_time"
        assert records[1].subject_key == "sub1"
        assert records[1].end_time == "E202604101450102711"
        assert records[1].has_required_training_fields is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _runtime_root(name: str) -> Path:
    root = Path("tests") / ".runtime" / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root

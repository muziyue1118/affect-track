import csv
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.models import SaveScoreRequest
from app.services.storage import CSVScoreStore
from app.services.video_catalog import VideoCatalog


def make_sandbox(name: str) -> Path:
    root = Path("tests") / ".runtime" / f"{name}_{uuid.uuid4().hex}"
    (root / "video").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    return root


def cleanup_sandbox(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def test_video_catalog_filters_and_sorts() -> None:
    root = make_sandbox("catalog")
    try:
        video_dir = root / "video"
        for name in ["neutral_2.mp4", "positive_1.mp4", "negative_1.mp4", "1.mp4", "positive_a.mp4"]:
            (video_dir / name).write_text("demo", encoding="utf-8")

        catalog = VideoCatalog(video_dir)
        items = catalog.list_videos()

        assert [item.name for item in items] == ["positive_1.mp4", "neutral_2.mp4", "negative_1.mp4"]
        assert catalog.category_for("positive_1.mp4") == "positive"
        assert catalog.category_for("1.mp4") is None
    finally:
        cleanup_sandbox(root)


def test_csv_store_creates_header_and_appends() -> None:
    root = make_sandbox("csv")
    try:
        store = CSVScoreStore(root / "data" / "offline_records.csv")
        payload = SaveScoreRequest(
            subject_id="SUBJ_001",
            video_name="positive_1.mp4",
            start_time="E202604091935251282",
            valence=4,
            arousal=3,
        )

        store.save_score(payload, "positive")
        store.save_score(payload, "positive")

        with (root / "data" / "offline_records.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2
        assert rows[0]["subject_id"] == "SUBJ_001"
        assert rows[0]["category"] == "positive"
    finally:
        cleanup_sandbox(root)


def test_csv_store_is_thread_safe() -> None:
    root = make_sandbox("threadsafe")
    try:
        store = CSVScoreStore(root / "data" / "offline_records.csv")
        payload = SaveScoreRequest(
            subject_id="SUBJ_001",
            video_name="positive_1.mp4",
            start_time="E202604091935251282",
            valence=4,
            arousal=3,
        )

        def write_once(_: int) -> None:
            store.save_score(payload, "positive")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write_once, range(20)))

        with (root / "data" / "offline_records.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 20
    finally:
        cleanup_sandbox(root)

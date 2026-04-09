import csv
from datetime import datetime
from pathlib import Path
from threading import Lock

from app.models import SaveScoreRequest


CSV_HEADERS = [
    "subject_id",
    "video_name",
    "category",
    "start_time",
    "end_time",
    "valence",
    "arousal",
    "saved_at",
]


class CSVScoreStore:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def save_score(self, payload: SaveScoreRequest, category: str) -> str:
        saved_at = datetime.now().isoformat(timespec="milliseconds")
        row = {
            "subject_id": payload.subject_id,
            "video_name": payload.video_name,
            "category": category,
            "start_time": payload.start_time,
            "end_time": payload.end_time,
            "valence": payload.valence,
            "arousal": payload.arousal,
            "saved_at": saved_at,
        }
        with self._lock:
            file_exists = self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        return saved_at



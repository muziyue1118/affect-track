import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def make_runtime_root(name: str) -> Path:
    root = Path("tests") / ".runtime" / f"{name}_{uuid.uuid4().hex}"
    (root / "video").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    return root


def cleanup_runtime_root(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def test_get_videos_ignores_invalid_files() -> None:
    runtime_root = make_runtime_root("videos")
    try:
        (runtime_root / "video" / "positive_1.mp4").write_text("demo", encoding="utf-8")
        (runtime_root / "video" / "1.mp4").write_text("demo", encoding="utf-8")

        with TestClient(create_app(runtime_root)) as client:
            response = client.get("/api/videos")

        assert response.status_code == 200
        assert response.json() == [
            {
                "index": 1,
                "name": "positive_1.mp4",
                "category": "positive",
                "url": "/video/positive_1.mp4",
            }
        ]
    finally:
        cleanup_runtime_root(runtime_root)


def test_save_score_persists_csv() -> None:
    runtime_root = make_runtime_root("save_score")
    try:
        (runtime_root / "video" / "positive_1.mp4").write_text("demo", encoding="utf-8")

        with TestClient(create_app(runtime_root)) as client:
            response = client.post(
                "/api/save_score",
                json={
                    "subject_id": "SUBJ_001",
                    "video_name": "positive_1.mp4",
                    "start_time": "E202604091935251282",
                    "valence": 5,
                    "arousal": 4,
                },
            )

        assert response.status_code == 200
        csv_text = (runtime_root / "data" / "offline_records.csv").read_text(encoding="utf-8-sig")
        assert "subject_id,video_name,category,start_time,valence,arousal,saved_at" in csv_text
        assert "SUBJ_001,positive_1.mp4,positive,E202604091935251282,5,4" in csv_text
    finally:
        cleanup_runtime_root(runtime_root)


def test_save_score_rejects_invalid_payload() -> None:
    runtime_root = make_runtime_root("invalid_payload")
    try:
        (runtime_root / "video" / "positive_1.mp4").write_text("demo", encoding="utf-8")

        with TestClient(create_app(runtime_root)) as client:
            response = client.post(
                "/api/save_score",
                json={
                    "subject_id": "",
                    "video_name": "positive_1.mp4",
                    "start_time": "bad",
                    "valence": 9,
                    "arousal": 0,
                },
            )

        assert response.status_code == 422
    finally:
        cleanup_runtime_root(runtime_root)


def test_save_score_rejects_unknown_video() -> None:
    runtime_root = make_runtime_root("unknown_video")
    try:
        with TestClient(create_app(runtime_root)) as client:
            response = client.post(
                "/api/save_score",
                json={
                    "subject_id": "SUBJ_001",
                    "video_name": "positive_1.mp4",
                    "start_time": "E202604091935251282",
                    "valence": 5,
                    "arousal": 4,
                },
            )

        assert response.status_code == 400
    finally:
        cleanup_runtime_root(runtime_root)


def test_live_frame_broadcasts_to_websocket() -> None:
    runtime_root = make_runtime_root("live_frame")
    try:
        with TestClient(create_app(runtime_root)) as client:
            mode_response = client.post("/api/emotion_mode", json={"mode": "live"})
            assert mode_response.status_code == 200

            with client.websocket_connect("/ws/emotion_stream") as websocket:
                response = client.post(
                    "/api/emotion_frame",
                    json={
                        "timestamp": "19:35:25",
                        "valence": 3.45,
                        "arousal": 2.1,
                    },
                )
                assert response.status_code == 200
                payload = websocket.receive_json()

        assert payload["timestamp"] == "19:35:25"
        assert payload["valence"] == 3.45
        assert payload["arousal"] == 2.1
        assert payload["source"] == "live"
    finally:
        cleanup_runtime_root(runtime_root)

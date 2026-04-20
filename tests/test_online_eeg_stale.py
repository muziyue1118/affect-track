import asyncio
import time
from pathlib import Path

import pytest

from app.services.emotion_stream import EmotionStreamHub
from app.services.online_eeg import OnlineEmotionService, StaleEEGError


class DummyWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def test_live_frame_cache_expires_for_new_websocket_connections() -> None:
    async def scenario() -> None:
        hub = EmotionStreamHub()
        hub.live_frame_ttl_seconds = 0.01
        await hub.set_mode("live")
        await hub.publish({"timestamp": "19:35:25", "valence": 3.2, "arousal": 2.4}, source="live")
        await asyncio.sleep(0.03)

        websocket = DummyWebSocket()
        await hub.connect(websocket)  # type: ignore[arg-type]

        assert websocket.accepted is True
        assert websocket.sent == []

    asyncio.run(scenario())


def test_online_service_rejects_unchanged_sample_count() -> None:
    service = OnlineEmotionService(project_root=Path("."), stream_hub=EmotionStreamHub())
    service.state.sample_count = 4000
    service.state.last_data_at = time.time()
    service._last_sample_count = 4000

    with pytest.raises(StaleEEGError, match="sample_count did not increase"):
        service._assert_fresh({"connected": True})


def test_online_service_rejects_old_packets() -> None:
    service = OnlineEmotionService(project_root=Path("."), stream_hub=EmotionStreamHub())
    service.state.sample_count = 5000
    service.state.last_data_at = time.time() - 4.0

    with pytest.raises(StaleEEGError, match="last packet"):
        service._assert_fresh({"connected": True})


def test_online_service_waits_for_initial_window_without_stale_alarm() -> None:
    service = OnlineEmotionService(project_root=Path("."), stream_hub=EmotionStreamHub())
    service.state.sample_count = 1000
    service.state.last_data_at = time.time()

    with pytest.raises(RuntimeError, match="waiting for enough samples"):
        service._assert_fresh({"connected": True})

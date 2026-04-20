import asyncio
import math
import time
from datetime import datetime
from typing import Literal

from fastapi import WebSocket


EmotionMode = Literal["mock", "live"]


class EmotionStreamHub:
    def __init__(self) -> None:
        self.mode: EmotionMode = "mock"
        self.live_frame_ttl_seconds: float = 3.0
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._latest_by_source: dict[EmotionMode, dict] = {}
        self._latest_at_by_source: dict[EmotionMode, float] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
            latest_frame = self._latest_frame_if_fresh(self.mode)
        if latest_frame:
            await websocket.send_json(latest_frame)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def set_mode(self, mode: EmotionMode) -> None:
        self.mode = mode
        latest_frame = self._latest_frame_if_fresh(mode)
        if latest_frame:
            await self._broadcast(latest_frame)

    async def publish(self, frame: dict, source: EmotionMode) -> None:
        payload = {
            "timestamp": frame["timestamp"],
            "valence": round(float(frame["valence"]), 2),
            "arousal": round(float(frame["arousal"]), 2),
            "source": source,
        }
        self._latest_by_source[source] = payload
        self._latest_at_by_source[source] = time.monotonic()
        if self.mode == source:
            await self._broadcast(payload)

    def snapshot(self) -> dict:
        live_frame_age = self._frame_age_seconds("live")
        return {
            "mode": self.mode,
            "connected_clients": len(self._clients),
            "has_mock_frame": "mock" in self._latest_by_source,
            "has_live_frame": "live" in self._latest_by_source,
            "live_frame_age_seconds": live_frame_age,
            "live_frame_is_fresh": live_frame_age is not None and live_frame_age <= self.live_frame_ttl_seconds,
        }

    def _frame_age_seconds(self, source: EmotionMode) -> float | None:
        latest_at = self._latest_at_by_source.get(source)
        if latest_at is None:
            return None
        return round(time.monotonic() - latest_at, 3)

    def _latest_frame_if_fresh(self, source: EmotionMode) -> dict | None:
        latest_frame = self._latest_by_source.get(source)
        if latest_frame is None:
            return None
        if source == "live":
            latest_at = self._latest_at_by_source.get(source)
            if latest_at is None or time.monotonic() - latest_at > self.live_frame_ttl_seconds:
                return None
        return latest_frame

    async def _broadcast(self, payload: dict) -> None:
        async with self._lock:
            clients = list(self._clients)

        stale_clients: list[WebSocket] = []
        for client in clients:
            try:
                await client.send_json(payload)
            except Exception:
                stale_clients.append(client)

        if stale_clients:
            async with self._lock:
                for client in stale_clients:
                    self._clients.discard(client)


def build_mock_frame(elapsed_seconds: float) -> dict:
    # Keep the mock feed smooth and bounded between 1 and 5.
    valence = 3.0 + 1.15 * math.sin(elapsed_seconds / 2.8) + 0.2 * math.sin(elapsed_seconds / 0.9)
    arousal = 3.0 + 1.0 * math.cos(elapsed_seconds / 3.4) + 0.25 * math.sin(elapsed_seconds / 1.7)
    valence = min(5.0, max(1.0, valence))
    arousal = min(5.0, max(1.0, arousal))
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "valence": valence,
        "arousal": arousal,
    }


async def run_mock_stream(hub: EmotionStreamHub, tick_seconds: float = 0.1) -> None:
    start_time = time.perf_counter()
    while True:
        if hub.mode == "mock":
            frame = build_mock_frame(time.perf_counter() - start_time)
            await hub.publish(frame, source="mock")
        await asyncio.sleep(tick_seconds)

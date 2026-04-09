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
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._latest_by_source: dict[EmotionMode, dict] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
            latest_frame = self._latest_by_source.get(self.mode)
        if latest_frame:
            await websocket.send_json(latest_frame)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def set_mode(self, mode: EmotionMode) -> None:
        self.mode = mode
        latest_frame = self._latest_by_source.get(mode)
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
        if self.mode == source:
            await self._broadcast(payload)

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "connected_clients": len(self._clients),
            "has_mock_frame": "mock" in self._latest_by_source,
            "has_live_frame": "live" in self._latest_by_source,
        }

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

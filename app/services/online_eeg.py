from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from analysis.Net import build_torch_model
from analysis.config import load_config
from analysis.online_preprocessing import preprocess_online_eeg_window, probability_to_score
from app.services.emotion_stream import EmotionStreamHub


@dataclass
class OnlineEEGSettings:
    host: str = "127.0.0.1"
    port: int = 8712
    channels: int = 32
    srate: int = 1000
    window_seconds: float = 4.0
    tick_seconds: float = 1.0
    smoothing_alpha: float = 0.25
    stale_after_seconds: float = 2.5
    live_frame_ttl_seconds: float = 3.0
    stale_log_interval_seconds: float = 10.0


@dataclass
class OnlineEEGState:
    running: bool = False
    model_ready: bool = False
    status: str = "idle"
    message: str = "Online EEG service is idle."
    last_frame: dict | None = None
    last_error: str | None = None
    last_data_at: float | None = None
    last_publish_at: float | None = None
    stale_seconds: float | None = None
    sample_count: int | None = None
    consecutive_stale_count: int = 0
    started_at: float | None = None
    settings: OnlineEEGSettings = field(default_factory=OnlineEEGSettings)


class StaleEEGError(RuntimeError):
    pass


class OnlineEmotionService:
    def __init__(
        self,
        *,
        project_root: Path,
        stream_hub: EmotionStreamHub,
        config_path: Path | None = None,
        model_dir: Path | None = None,
        device: str = "auto",
        recorder_factory: Callable[..., object] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.project_root = project_root
        self.stream_hub = stream_hub
        self.config_path = config_path or project_root / "analysis" / "eeg_config.yaml"
        self.model_dir = model_dir or project_root / "models" / "emotion_online"
        self.device_request = device
        self.recorder_factory = recorder_factory
        self.logger = logger or logging.getLogger("online-eeg")
        self.settings = OnlineEEGSettings()
        self.stream_hub.live_frame_ttl_seconds = self.settings.live_frame_ttl_seconds
        self.state = OnlineEEGState(settings=self.settings)
        self._task: asyncio.Task | None = None
        self._recorder = None
        self._models: dict[str, object] = {}
        self._metadata: dict | None = None
        self._device = None
        self._smoothed: dict[str, float] = {}
        self._last_sample_count: int | None = None
        self._was_stale = False
        self._last_stale_log_at = 0.0

    async def start(self) -> dict:
        if self.state.running:
            return self.snapshot()
        self._load_models()
        if not self.state.model_ready:
            return self.snapshot()
        try:
            self._recorder = self._build_recorder()
            await asyncio.to_thread(self._recorder.start)
        except Exception as exc:  # pragma: no cover - depends on live EEG hardware.
            self.state.status = "device_error"
            self.state.message = f"Failed to connect EEG device: {exc}"
            self.state.last_error = str(exc)
            self.logger.exception("failed to start online EEG recorder")
            return self.snapshot()
        self.state.running = True
        self.state.started_at = time.time()
        self.state.status = "running"
        self.state.message = "Online EEG service is running."
        self.state.last_data_at = None
        self.state.last_publish_at = None
        self.state.stale_seconds = None
        self.state.sample_count = None
        self.state.consecutive_stale_count = 0
        self._last_sample_count = None
        self._was_stale = False
        self._last_stale_log_at = 0.0
        self._task = asyncio.create_task(self._run_loop())
        return self.snapshot()

    async def stop(self) -> dict:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._recorder is not None:
            try:
                await asyncio.to_thread(self._recorder.stop)
            except Exception as exc:  # pragma: no cover - best effort shutdown.
                self.logger.warning("failed to stop EEG recorder cleanly: %s", exc)
        self._recorder = None
        self.state.running = False
        self.state.status = "idle"
        self.state.message = "Online EEG service is stopped."
        self.state.stale_seconds = None
        self.state.consecutive_stale_count = 0
        self._last_sample_count = None
        self._was_stale = False
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "running": self.state.running,
            "model_ready": self.state.model_ready,
            "status": self.state.status,
            "message": self.state.message,
            "last_frame": self.state.last_frame,
            "last_error": self.state.last_error,
            "last_data_at": self.state.last_data_at,
            "last_publish_at": self.state.last_publish_at,
            "stale_seconds": self.state.stale_seconds,
            "sample_count": self.state.sample_count,
            "consecutive_stale_count": self.state.consecutive_stale_count,
            "started_at": self.state.started_at,
            "settings": self.settings.__dict__,
            "model_dir": str(self.model_dir),
            "device": str(self._device) if self._device is not None else self.device_request,
        }

    def _load_models(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.state.model_ready = False
            self.state.status = "model_missing"
            self.state.message = "PyTorch is not installed; online EEG inference cannot start."
            self.state.last_error = str(exc)
            return
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            self.state.model_ready = False
            self.state.status = "model_missing"
            self.state.message = f"Online model metadata not found: {metadata_path}"
            return
        try:
            import json

            self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._device = _resolve_device(self.device_request, torch)
            network = str(self._metadata["network"])
            self._models = {}
            for task in ("valence", "arousal"):
                artifact = self.model_dir / str(self._metadata["artifacts"][task])
                checkpoint = torch.load(artifact, map_location=self._device, weights_only=False)
                model = build_torch_model(
                    network,
                    n_channels=int(checkpoint["n_channels"]),
                    n_classes=1,
                    n_samples=int(checkpoint["n_samples"]),
                ).to(self._device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self._models[task] = model
            self.state.model_ready = True
            self.state.status = "ready"
            self.state.message = "Online EEG models are loaded."
            self.state.last_error = None
        except Exception as exc:
            self._models = {}
            self.state.model_ready = False
            self.state.status = "model_error"
            self.state.message = f"Failed to load online EEG models: {exc}"
            self.state.last_error = str(exc)
            self.logger.exception("failed to load online EEG models")

    def _build_recorder(self):
        if self.recorder_factory is not None:
            return self.recorder_factory(
                n_chan=self.settings.channels,
                srate=self.settings.srate,
                host=self.settings.host,
                port=self.settings.port,
                t_buffer=max(10, int(self.settings.window_seconds * 3)),
            )
        from online.record.eeg_recorder_utils import EEGRecorder

        return EEGRecorder(
            n_chan=self.settings.channels,
            srate=self.settings.srate,
            host=self.settings.host,
            port=self.settings.port,
            t_buffer=max(10, int(self.settings.window_seconds * 3)),
        )

    async def _run_loop(self) -> None:
        config = load_config(self.config_path)
        while True:
            try:
                freshness = self._get_recorder_freshness()
                self._assert_fresh(freshness)
                raw_window = await asyncio.to_thread(self._recorder.get_record, self.settings.window_seconds)
                processed = preprocess_online_eeg_window(
                    raw_window,
                    input_sfreq=self.settings.srate,
                    config=config,
                    expected_channels=self.settings.channels,
                )
                scores = self._predict_scores(processed)
                frame = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "valence": scores["valence"],
                    "arousal": scores["arousal"],
                }
                await self.stream_hub.publish(frame, source="live")
                self.state.last_frame = {**frame, "source": "live"}
                self.state.last_publish_at = time.time()
                self.state.status = "running"
                self.state.message = "Online EEG frame published."
                self.state.last_error = None
                self.state.stale_seconds = 0.0
                self.state.consecutive_stale_count = 0
                self._last_sample_count = self.state.sample_count
                if self._was_stale:
                    self.logger.info("online_eeg_recovered sample_count=%s", self.state.sample_count)
                self._was_stale = False
            except StaleEEGError as exc:
                self.state.status = "stale_eeg"
                self.state.message = f"EEG data stream is stale; waiting for new samples: {exc}"
                self.state.last_error = str(exc)
                self.state.consecutive_stale_count += 1
                self._log_stale_event(str(exc))
            except Exception as exc:
                self.state.status = "waiting_for_valid_eeg"
                self.state.message = f"Waiting for valid EEG: {exc}"
                self.state.last_error = str(exc)
            await asyncio.sleep(self.settings.tick_seconds)

    def _get_recorder_freshness(self) -> dict:
        if self._recorder is None:
            raise StaleEEGError("recorder is not initialized")
        if hasattr(self._recorder, "get_freshness_info"):
            freshness = self._recorder.get_freshness_info()
        elif hasattr(self._recorder, "server") and hasattr(self._recorder.server, "GetFreshnessInfo"):
            freshness = self._recorder.server.GetFreshnessInfo()
        else:
            freshness = {}
        sample_count = freshness.get("sample_count")
        if sample_count is not None:
            sample_count = int(sample_count)
        last_packet_at = freshness.get("last_packet_at")
        self.state.sample_count = sample_count
        self.state.last_data_at = float(last_packet_at) if last_packet_at is not None else None
        if self.state.last_data_at is None:
            self.state.stale_seconds = None
        else:
            self.state.stale_seconds = round(max(0.0, time.time() - self.state.last_data_at), 3)
        return freshness

    def _assert_fresh(self, freshness: dict) -> None:
        sample_count = self.state.sample_count
        last_packet_at = self.state.last_data_at
        connected = freshness.get("connected")
        recorder_error = freshness.get("last_error")
        expected_samples = int(self.settings.window_seconds * self.settings.srate)
        now = time.time()

        if connected is False:
            raise StaleEEGError(f"device disconnected ({recorder_error or 'socket not connected'})")
        if sample_count is None:
            raise StaleEEGError("recorder does not expose sample_count")
        if last_packet_at is None:
            elapsed_since_start = now - self.state.started_at if self.state.started_at is not None else 0.0
            if elapsed_since_start <= self.settings.stale_after_seconds:
                raise RuntimeError("waiting for first EEG packet")
            raise StaleEEGError("no EEG packets have been received")
        stale_seconds = now - last_packet_at
        self.state.stale_seconds = round(max(0.0, stale_seconds), 3)
        if stale_seconds > self.settings.stale_after_seconds:
            raise StaleEEGError(
                f"last packet is {stale_seconds:.2f}s old; threshold is {self.settings.stale_after_seconds:.2f}s"
            )
        if sample_count < expected_samples:
            raise RuntimeError(f"waiting for enough samples ({sample_count}/{expected_samples})")
        if self._last_sample_count is not None and sample_count <= self._last_sample_count:
            raise StaleEEGError(
                f"sample_count did not increase ({sample_count} <= {self._last_sample_count})"
            )

    def _log_stale_event(self, reason: str) -> None:
        now = time.time()
        event_name = (
            "online_eeg_device_disconnected"
            if reason.startswith("device disconnected")
            else "online_eeg_stale_detected"
        )
        if not self._was_stale:
            self.logger.warning(
                "%s reason=%s sample_count=%s stale_seconds=%s",
                event_name,
                reason,
                self.state.sample_count,
                self.state.stale_seconds,
            )
            self._was_stale = True
            self._last_stale_log_at = now
            return
        if now - self._last_stale_log_at >= self.settings.stale_log_interval_seconds:
            self.logger.warning(
                "%s reason=%s sample_count=%s stale_seconds=%s consecutive=%s",
                event_name,
                reason,
                self.state.sample_count,
                self.state.stale_seconds,
                self.state.consecutive_stale_count,
            )
            self._last_stale_log_at = now

    def _predict_scores(self, processed_window) -> dict[str, float]:
        import torch

        tensor = torch.tensor(processed_window[None, :, :], dtype=torch.float32, device=self._device)
        scores: dict[str, float] = {}
        with torch.no_grad():
            for task, model in self._models.items():
                output = model(tensor)
                if isinstance(output, dict):
                    output = output["logits"]
                if isinstance(output, tuple):
                    output = output[0]
                probability = float(torch.sigmoid(output.reshape(-1)[0]).detach().cpu().item())
                score = probability_to_score(probability)
                previous = self._smoothed.get(task)
                if previous is not None:
                    alpha = self.settings.smoothing_alpha
                    score = alpha * score + (1.0 - alpha) * previous
                self._smoothed[task] = score
                scores[task] = round(score, 3)
        return scores


def _resolve_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA in this environment.")
    return resolved

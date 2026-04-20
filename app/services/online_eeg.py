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


@dataclass
class OnlineEEGState:
    running: bool = False
    model_ready: bool = False
    status: str = "idle"
    message: str = "Online EEG service is idle."
    last_frame: dict | None = None
    last_error: str | None = None
    started_at: float | None = None
    settings: OnlineEEGSettings = field(default_factory=OnlineEEGSettings)


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
        self.state = OnlineEEGState(settings=self.settings)
        self._task: asyncio.Task | None = None
        self._recorder = None
        self._models: dict[str, object] = {}
        self._metadata: dict | None = None
        self._device = None
        self._smoothed: dict[str, float] = {}

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
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "running": self.state.running,
            "model_ready": self.state.model_ready,
            "status": self.state.status,
            "message": self.state.message,
            "last_frame": self.state.last_frame,
            "last_error": self.state.last_error,
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
                self.state.status = "running"
                self.state.message = "Online EEG frame published."
                self.state.last_error = None
            except Exception as exc:
                self.state.status = "waiting_for_valid_eeg"
                self.state.message = f"Waiting for valid EEG: {exc}"
                self.state.last_error = str(exc)
            await asyncio.sleep(self.settings.tick_seconds)

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

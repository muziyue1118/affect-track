import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models import (
    EmotionFrame,
    EmotionModeRequest,
    EmotionModeResponse,
    SaveScoreRequest,
    SaveScoreResponse,
    VideoItem,
)
from app.services.emotion_stream import EmotionStreamHub, run_mock_stream
from app.services.storage import CSVScoreStore
from app.services.video_catalog import VideoCatalog
from app.utils.timestamps import generate_timestamp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S"),
        }
        return json.dumps(message, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def create_app(runtime_root: Path | None = None) -> FastAPI:
    configure_logging()
    runtime_root = runtime_root or PROJECT_ROOT
    data_dir = runtime_root / "data"
    video_dir = runtime_root / "video"
    data_dir.mkdir(parents=True, exist_ok=True)

    catalog = VideoCatalog(video_dir)
    score_store = CSVScoreStore(data_dir / "offline_records.csv")
    stream_hub = EmotionStreamHub()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    logger = logging.getLogger("emotion-app")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("starting mock emotion stream")
        mock_task = asyncio.create_task(run_mock_stream(stream_hub))
        try:
            yield
        finally:
            mock_task.cancel()
            with suppress(asyncio.CancelledError):
                await mock_task
            logger.info("stopped mock emotion stream")

    app = FastAPI(
        title="Emotion Induction System",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.video_catalog = catalog
    app.state.score_store = score_store
    app.state.stream_hub = stream_hub
    app.state.logger = logger

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/video", StaticFiles(directory=str(video_dir), check_dir=False), name="video")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/offline")

    @app.get("/offline", response_class=HTMLResponse, include_in_schema=False)
    async def offline_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="offline.html",
            context={"page_title": "离线情绪采集"},
        )

    @app.get("/online", response_class=HTMLResponse, include_in_schema=False)
    async def online_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="online.html",
            context={"page_title": "在线情绪演示"},
        )

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "timestamp": generate_timestamp(),
            "videos_available": len(catalog.list_videos()),
            **stream_hub.snapshot(),
        }

    @app.get("/api/videos", response_model=list[VideoItem])
    async def get_videos() -> list[VideoItem]:
        return catalog.list_videos()

    @app.get("/api/emotion_mode", response_model=EmotionModeResponse)
    async def get_emotion_mode() -> EmotionModeResponse:
        return EmotionModeResponse(mode=stream_hub.mode)

    @app.post("/api/emotion_mode", response_model=EmotionModeResponse)
    async def set_emotion_mode(payload: EmotionModeRequest) -> EmotionModeResponse:
        await stream_hub.set_mode(payload.mode)
        return EmotionModeResponse(mode=payload.mode)

    @app.post("/api/save_score", response_model=SaveScoreResponse)
    async def save_score(payload: SaveScoreRequest) -> SaveScoreResponse:
        category = catalog.category_for(payload.video_name)
        if category is None:
            raise HTTPException(status_code=400, detail="Unknown or unavailable video_name.")
        try:
            saved_at = score_store.save_score(payload, category)
        except OSError as exc:
            logger.exception("failed to save score")
            raise HTTPException(status_code=500, detail="Failed to persist score.") from exc
        logger.info("saved score for %s / %s", payload.subject_id, payload.video_name)
        return SaveScoreResponse(status="ok", saved_at=saved_at)

    @app.post("/api/emotion_frame")
    async def ingest_emotion_frame(payload: EmotionFrame) -> dict:
        await stream_hub.publish(payload.model_dump(), source="live")
        logger.info("accepted live emotion frame")
        return {"status": "ok", "mode": stream_hub.mode}

    @app.websocket("/ws/emotion_stream")
    async def emotion_stream(websocket: WebSocket) -> None:
        await stream_hub.connect(websocket)
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            await stream_hub.disconnect(websocket)

    return app


app = create_app()

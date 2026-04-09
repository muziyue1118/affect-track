from typing import Literal

from pydantic import BaseModel, Field


TIMESTAMP_PATTERN = r"^E\d{18}$"
SUBJECT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
VIDEO_NAME_PATTERN = r"^(positive|neutral|negative)_(\d+)\.mp4$"


class VideoItem(BaseModel):
    index: int = Field(..., ge=1)
    name: str = Field(..., pattern=VIDEO_NAME_PATTERN)
    category: Literal["positive", "neutral", "negative"]
    url: str = Field(..., min_length=1)


class SaveScoreRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=64, pattern=SUBJECT_ID_PATTERN)
    video_name: str = Field(..., pattern=VIDEO_NAME_PATTERN)
    start_time: str = Field(..., pattern=TIMESTAMP_PATTERN)
    valence: int = Field(..., ge=1, le=5)
    arousal: int = Field(..., ge=1, le=5)


class SaveScoreResponse(BaseModel):
    status: Literal["ok"]
    saved_at: str = Field(..., min_length=1)


class EmotionFrame(BaseModel):
    timestamp: str = Field(..., min_length=1, max_length=32)
    valence: float = Field(..., ge=1.0, le=5.0)
    arousal: float = Field(..., ge=1.0, le=5.0)
    source: Literal["mock", "live"] = "live"


class EmotionModeRequest(BaseModel):
    mode: Literal["mock", "live"]


class EmotionModeResponse(BaseModel):
    mode: Literal["mock", "live"]

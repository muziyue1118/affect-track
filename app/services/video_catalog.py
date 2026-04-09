import re
from pathlib import Path

from app.models import VideoItem


VIDEO_PATTERN = re.compile(r"^(positive|neutral|negative)_(\d+)\.mp4$")
CATEGORY_ORDER = {"positive": 0, "neutral": 1, "negative": 2}


class VideoCatalog:
    def __init__(self, video_dir: Path) -> None:
        self.video_dir = video_dir

    def list_videos(self) -> list[VideoItem]:
        items: list[VideoItem] = []
        if not self.video_dir.exists():
            return items

        matched_files: list[tuple[str, int, str]] = []
        for path in self.video_dir.iterdir():
            if not path.is_file():
                continue
            match = VIDEO_PATTERN.fullmatch(path.name)
            if not match:
                continue
            category = match.group(1)
            ordinal = int(match.group(2))
            matched_files.append((category, ordinal, path.name))

        matched_files.sort(key=lambda item: (CATEGORY_ORDER[item[0]], item[1], item[2]))
        for index, (category, _ordinal, file_name) in enumerate(matched_files, start=1):
            items.append(
                VideoItem(
                    index=index,
                    name=file_name,
                    category=category,
                    url=f"/video/{file_name}",
                )
            )
        return items

    def category_for(self, video_name: str) -> str | None:
        match = VIDEO_PATTERN.fullmatch(video_name)
        if not match:
            return None
        if not (self.video_dir / video_name).exists():
            return None
        return match.group(1)

    def is_known_video(self, video_name: str) -> bool:
        return self.category_for(video_name) is not None

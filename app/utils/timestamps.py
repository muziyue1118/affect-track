import re
from datetime import datetime


TIMESTAMP_REGEX = re.compile(r"^E\d{18}$")


def generate_timestamp(now: datetime | None = None) -> str:
    """Generate E + YYYYMMDDHHMMSS + first four digits of microseconds."""
    now = now or datetime.now()
    time_str = now.strftime("%Y%m%d%H%M%S")
    micro_str = now.strftime("%f")[:4]
    return f"E{time_str}{micro_str}"


def validate_timestamp(value: str) -> bool:
    return bool(TIMESTAMP_REGEX.fullmatch(value))

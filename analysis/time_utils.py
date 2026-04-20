from __future__ import annotations

from datetime import datetime


TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


def parse_e_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or len(value) != 19 or not value.startswith("E"):
        raise ValueError(f"Invalid E timestamp: {value!r}")
    main = value[1:15]
    subsecond = value[15:]
    if not main.isdigit() or not subsecond.isdigit():
        raise ValueError(f"Invalid E timestamp: {value!r}")
    parsed = datetime.strptime(main, TIMESTAMP_FORMAT)
    return parsed.replace(microsecond=int(subsecond) * 100)


def is_e_timestamp(value: str | None) -> bool:
    if value is None:
        return False
    try:
        parse_e_timestamp(value)
    except ValueError:
        return False
    return True


def format_run_timestamp(now: datetime | None = None) -> str:
    value = now or datetime.now()
    return value.strftime("%Y%m%d_%H%M%S")

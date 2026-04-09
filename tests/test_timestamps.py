from datetime import datetime

from app.utils.timestamps import generate_timestamp, validate_timestamp


def test_generate_timestamp_matches_required_format() -> None:
    fixed = datetime(2026, 4, 9, 19, 35, 25, 128256)
    assert generate_timestamp(fixed) == "E202604091935251282"


def test_validate_timestamp() -> None:
    assert validate_timestamp("E202604091935251282") is True
    assert validate_timestamp("E20260409193525128") is False
    assert validate_timestamp("202604091935251282") is False

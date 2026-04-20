from datetime import datetime
import shutil
import uuid
from pathlib import Path

from analysis.bdf import read_bdf_header


def test_read_bdf_header_minimal_file() -> None:
    root = _runtime_root("bdf")
    try:
        bdf = root / "data.bdf"
        _write_minimal_bdf(bdf)

        header = read_bdf_header(bdf)

        assert header.n_signals == 2
        assert header.labels == ("Fp1", "Fp2")
        assert header.sample_rate_hz == 1000
        assert header.duration_s == 10
        assert header.start_time == datetime(2026, 4, 10, 16, 7, 40)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_minimal_bdf(path: Path) -> None:
    n_signals = 2
    fixed = (
        _field("BIOSEMI", 8)
        + _field("sub1", 80)
        + _field("recording", 80)
        + _field("10.04.26", 8)
        + _field("16.07.40", 8)
        + _field(str(256 + 256 * n_signals), 8)
        + _field("", 44)
        + _field("10", 8)
        + _field("1", 8)
        + _field(str(n_signals), 4)
    )
    labels = _field("Fp1", 16) + _field("Fp2", 16)
    transducer = _field("", 80) * n_signals
    physical_dim = _field("uV", 8) * n_signals
    physical_min = _field("-375000", 8) * n_signals
    physical_max = _field("375000", 8) * n_signals
    digital_min = _field("-8388608", 8) * n_signals
    digital_max = _field("8388607", 8) * n_signals
    prefilter = _field("HP: DC, LP: 250", 80) * n_signals
    samples = _field("1000", 8) * n_signals
    reserved = _field("", 32) * n_signals
    path.write_bytes(
        fixed
        + labels
        + transducer
        + physical_dim
        + physical_min
        + physical_max
        + digital_min
        + digital_max
        + prefilter
        + samples
        + reserved
    )


def _field(value: str, width: int) -> bytes:
    return value.encode("ascii").ljust(width, b" ")[:width]


def _runtime_root(name: str) -> Path:
    root = Path("tests") / ".runtime" / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root

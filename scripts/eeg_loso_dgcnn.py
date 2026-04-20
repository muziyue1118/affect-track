from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.eeg_pipeline import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "run",
                "--config",
                str(ROOT / "analysis" / "eeg_config.yaml"),
                "--task",
                "category",
                "--split-mode",
                "loso",
                "--model",
                "deep",
                "--deep-network",
                "DGCNN",
                "--protocol",
                "supervised",
                "--input-kind",
                "de",
            ]
        )
    )

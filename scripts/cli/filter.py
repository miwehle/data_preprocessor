from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ipy.io import dataset_path
from ipy.ops import filter


def main() -> int:
    filter(
        input_path=dataset_path("europarl", "normalized", "europarl_de-en_train.norm.jsonl"),
        output_path=dataset_path("europarl", "filtered", "europarl_de-en_train.filtered.jsonl"),
        flaw_report_path=dataset_path("europarl", "reports", "flaw_report.txt"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


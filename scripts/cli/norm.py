from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ipy.io import dataset_path
from ipy.ops import norm


def main() -> int:
    norm(
        input_path=dataset_path("europarl", "raw", "europarl_de-en_train.jsonl"),
        output_path=dataset_path("europarl", "normalized", "europarl_de-en_train.norm.jsonl"),
        norm_report_path=dataset_path("europarl", "reports", "norm_report.txt"),
        norm_debug=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

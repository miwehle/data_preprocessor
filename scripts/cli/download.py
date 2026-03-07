from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ipy.io import dataset_path
from ipy.ops import download


def main() -> int:
    download(
        dataset="Helsinki-NLP/europarl",
        config="de-en",
        split="train",
        output=dataset_path("europarl", "raw", "europarl_de-en_train.jsonl"),
        max_records=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ipy.io import dataset_path
from ipy.ops import tokenize


def main() -> int:
    tokenize(
        input_path=dataset_path("europarl", "filtered", "europarl_de-en_train.filtered.jsonl"),
        output_path=dataset_path("europarl", "tokenized", "europarl_de-en_train.tokenized.jsonl"),
        tokenize_report_path=dataset_path("europarl", "reports", "tokenize_report.txt"),
        model_name="Helsinki-NLP/opus-mt-de-en",
        tokenizer_kwargs={"truncation": True, "max_length": 256},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

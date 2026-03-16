from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


NUM_EXAMPLES = 1000
OUT_PATH = Path("tests/data/testdata_de_en_1000.jsonl")


def main() -> None:
    ds = load_dataset("Helsinki-NLP/europarl", "de-en", split=f"train[:{NUM_EXAMPLES}]")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(NUM_EXAMPLES):
            t = ds[i]["translation"]
            row = {"translation": {"de": t["de"], "en": t["en"]}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved {NUM_EXAMPLES} pairs to: {OUT_PATH}")


if __name__ == "__main__":
    main()

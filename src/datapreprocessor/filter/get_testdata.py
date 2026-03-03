from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


OUT_PATH = Path("data/testdata_de_en_100.jsonl")


def main() -> None:
    ds = load_dataset("Helsinki-NLP/europarl", "de-en", split="train")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(100):
            t = ds[i]["translation"]
            row = {"translation": {"de": t["de"], "en": t["en"]}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved 100 pairs to: {OUT_PATH}")


if __name__ == "__main__":
    main()

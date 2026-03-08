from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from datapreprocessor.types import Example


def dataset_path(dataset: str, stage: str, filename: str) -> Path:
    def repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    return repo_root() / "data" / dataset / stage / filename


def load(path: str | Path):
    def is_jsonl_path(input_path: Path) -> bool:
        return input_path.suffix.lower() == ".jsonl"

    def load_jsonl(input_path: Path):
        from datasets import load_dataset

        return load_dataset("json", data_files=str(input_path), split="train")

    from datasets import load_from_disk

    dataset_path = Path(path)
    if is_jsonl_path(dataset_path):
        return load_jsonl(dataset_path)
    return load_from_disk(str(dataset_path))


def save(examples: Iterable[Example], output_path: str | Path) -> None:
    def is_jsonl_path(output: Path) -> bool:
        return output.suffix.lower() == ".jsonl"

    def write_jsonl(examples: Iterable[Example], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from datasets import Dataset

    out_path = Path(output_path)
    if is_jsonl_path(out_path):
        write_jsonl(examples, out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = examples if isinstance(examples, Dataset) else Dataset.from_list(list(examples))
    ds.save_to_disk(str(out_path))

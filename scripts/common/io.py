from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

Example = dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dataset_path(dataset: str, stage: str, filename: str) -> Path:
    return repo_root() / "data" / dataset / stage / filename


def load_jsonl(path: str | Path):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split="train")


def write_jsonl(examples: Iterable[Example], output_path: str | Path) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _is_jsonl_path(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"


def load_examples(path: str | Path):
    from datasets import load_from_disk

    dataset_path = Path(path)
    if _is_jsonl_path(dataset_path):
        return load_jsonl(dataset_path)
    return load_from_disk(str(dataset_path))


def write_examples(examples: Iterable[Example], output_path: str | Path) -> None:
    from datasets import Dataset

    out_path = Path(output_path)
    if _is_jsonl_path(out_path):
        write_jsonl(examples, out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_generator(lambda: examples)
    ds.save_to_disk(str(out_path))

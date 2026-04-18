from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from data_preprocessor.io import load, save
from data_preprocessor.shared import SplitConfig

_DATASET_MANIFEST = "dataset_manifest.yaml"
_SPLIT_MANIFEST = "split_manifest.yaml"
_RATIO_TOLERANCE = 1e-9


def _validate_split_config(config: SplitConfig) -> Path:
    if config.dataset is None:
        raise ValueError("split_config.dataset is required.")
    if len(config.split_ratio) < 2:
        raise ValueError("split_config.split_ratio must define at least two splits.")
    if any(ratio < 0 for ratio in config.split_ratio.values()):
        raise ValueError("split_config.split_ratio values must be >= 0.")
    if abs(sum(config.split_ratio.values()) - 1.0) > _RATIO_TOLERANCE:
        raise ValueError("split_config.split_ratio must sum to 1.0.")
    dataset_dir = Path(config.dataset)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    if not (dataset_dir / _DATASET_MANIFEST).is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_dir / _DATASET_MANIFEST}")
    return dataset_dir


def _split_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    assigned = 0
    cumulative = 0.0
    items = list(ratios.items())
    for i, (name, ratio) in enumerate(items):
        if i == len(items) - 1:
            counts[name] = total - assigned
            break
        cumulative += ratio
        next_assigned = round(total * cumulative)
        counts[name] = next_assigned - assigned
        assigned = next_assigned
    return counts


def _split_output_dir(dataset_dir: Path, split_name: str) -> Path:
    return dataset_dir.with_name(f"{dataset_dir.name}_split-{split_name}")


def _write_split_manifest(output_dir: Path, dataset_dir: Path, split_name: str, config: SplitConfig, count: int) -> None:
    manifest = {
        "source_dataset": str(dataset_dir),
        "split_name": split_name,
        "split_seed": config.seed,
        "split_ratio": dict(config.split_ratio),
        "num_examples": count,
    }
    (output_dir / _SPLIT_MANIFEST).write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def split_dataset(config: SplitConfig) -> None:
    dataset_dir = _validate_split_config(config)
    dataset_manifest = yaml.safe_load((dataset_dir / _DATASET_MANIFEST).read_text(encoding="utf-8"))
    dataset = load(dataset_dir)
    shuffled = dataset.shuffle(seed=config.seed)
    counts = _split_counts(len(shuffled), config.split_ratio)
    start = 0
    for split_name, count in counts.items():
        output_dir = _split_output_dir(dataset_dir, split_name)
        if output_dir.exists():
            raise FileExistsError(f"Split output already exists: {output_dir}")
        subset = shuffled.select(range(start, start + count))
        save(subset, output_dir)
        split_dataset_manifest = dict(dataset_manifest)
        split_dataset_manifest["num_examples"] = count
        (output_dir / _DATASET_MANIFEST).write_text(
            yaml.safe_dump(split_dataset_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        _write_split_manifest(output_dir, dataset_dir, split_name, config, count)
        start += count

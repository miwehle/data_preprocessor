from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from datasets import Dataset, load_from_disk
import yaml

from data_preprocessor import api


def _dataset_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / ".local_tmp" / "tests" / uuid4().hex / "dataset"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_dataset(dataset_dir: Path, rows: list[dict], manifest: dict[str, object]) -> None:
    Dataset.from_list(rows).save_to_disk(str(dataset_dir))
    (dataset_dir / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def _read_manifest(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_split_writes_split_dirs_and_manifests():
    dataset_dir = _dataset_dir()
    manifest = {
        "schema_version": 1,
        "tokenizer_model_name": "test-tokenizer",
        "src_lang": "de",
        "tgt_lang": "en",
        "id_field": "id",
        "src_field": "src_ids",
        "tgt_field": "tgt_ids",
        "base_vocab_size": 32,
        "src_vocab_size": 33,
        "tgt_vocab_size": 34,
        "src_pad_id": 30,
        "tgt_pad_id": 31,
        "tgt_bos_id": 32,
        "tgt_eos_id": 2,
        "num_examples": 5,
        "configured_max_seq_len": 128,
    }
    rows = [{"id": i, "src_ids": [i, i + 1], "tgt_ids": [32, i + 10, 2]} for i in range(5)]
    _write_dataset(dataset_dir, rows, manifest)

    api.split(
        api.SplitConfig(
            dataset=str(dataset_dir), split_ratio={"train": 0.6, "val": 0.2, "test": 0.2}, seed=7
        )
    )

    train_dir = dataset_dir.with_name(f"{dataset_dir.name}_split-train")
    val_dir = dataset_dir.with_name(f"{dataset_dir.name}_split-val")
    test_dir = dataset_dir.with_name(f"{dataset_dir.name}_split-test")
    assert len(load_from_disk(str(train_dir))) == 3
    assert len(load_from_disk(str(val_dir))) == 1
    assert len(load_from_disk(str(test_dir))) == 1

    train_manifest = _read_manifest(train_dir / "dataset_manifest.yaml")
    assert train_manifest == {**manifest, "num_examples": 3}
    train_split_manifest = _read_manifest(train_dir / "split_manifest.yaml")
    assert train_split_manifest == {
        "source_dataset": str(dataset_dir),
        "split_name": "train",
        "split_seed": 7,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "num_examples": 3,
    }


def test_split_requires_dataset_path():
    try:
        api.split(api.SplitConfig(split_ratio={"train": 0.9, "val": 0.1}, seed=1))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "dataset" in str(exc)

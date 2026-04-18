from __future__ import annotations

import importlib
import json
from pathlib import Path
from uuid import uuid4

from datasets import Dataset
from data_preprocessor import api

load_module = importlib.import_module("data_preprocessor.load")


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _make_out_path() -> Path:
    root = Path(__file__).resolve().parents[2] / ".local_tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{uuid4().hex}.jsonl"


def test_load_adds_ids_by_default(monkeypatch):
    ds = Dataset.from_list(
        [{"translation": {"de": "eins", "en": "one"}}, {"translation": {"de": "zwei", "en": "two"}}]
    )
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    api.load(api.LoadConfig(path_name="dummy", name="de-en", split="train"), out)

    rows = _read_jsonl(out)
    assert [row["id"] for row in rows] == [0, 1]


def test_load_can_disable_ids(monkeypatch):
    ds = Dataset.from_list([{"translation": {"de": "eins", "en": "one"}}])
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    api.load(api.LoadConfig(path_name="dummy", name="de-en", split="train", include_ids=False), out)

    rows = _read_jsonl(out)
    assert "id" not in rows[0]


def test_load_raises_if_id_exists_and_overwrite_disabled(monkeypatch):
    ds = Dataset.from_list([{"id": 42, "translation": {"de": "eins", "en": "one"}}])
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    try:
        api.load(api.LoadConfig(path_name="dummy", name="de-en", split="train"), out)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "already exists" in str(exc)


def test_load_can_overwrite_existing_id(monkeypatch):
    ds = Dataset.from_list(
        [
            {"id": 42, "translation": {"de": "eins", "en": "one"}},
            {"id": 43, "translation": {"de": "zwei", "en": "two"}},
        ]
    )
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    api.load(api.LoadConfig(path_name="dummy", name="de-en", split="train", overwrite_ids=True, start_id=100), out)

    rows = _read_jsonl(out)
    assert [row["id"] for row in rows] == [100, 101]


def test_load_passes_load_dataset_args_through(monkeypatch):
    ds = Dataset.from_list([{"translation": {"de": "eins", "en": "one"}}])
    calls: list[tuple[tuple, dict]] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return ds

    monkeypatch.setattr(load_module, "load_dataset", fake_load_dataset)
    out = _make_out_path()

    api.load(api.LoadConfig(path_name="IWSLT/iwslt2017", name="iwslt2017-de-en", split="train"), out)

    rows = _read_jsonl(out)
    assert rows == [{"translation": {"de": "eins", "en": "one"}, "id": 0}]
    assert calls == [
        (
            (),
            {"path": "IWSLT/iwslt2017", "name": "iwslt2017-de-en", "split": "train", "data_files": None},
        )
    ]


def test_load_passes_data_files_through(monkeypatch):
    ds = Dataset.from_list([{"translation": {"de": "eins", "en": "one"}}])
    calls: list[tuple[tuple, dict]] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return ds

    monkeypatch.setattr(load_module, "load_dataset", fake_load_dataset)
    out = _make_out_path()

    api.load(
        api.LoadConfig(
            path_name="parquet",
            split="train",
            data_files="https://huggingface.co/datasets/org/ds/resolve/rev/name/train.parquet",
        ),
        out,
    )

    rows = _read_jsonl(out)
    assert rows == [{"translation": {"de": "eins", "en": "one"}, "id": 0}]
    assert calls == [
        (
            (),
            {
                "path": "parquet",
                "name": None,
                "split": "train",
                "data_files": "https://huggingface.co/datasets/org/ds/resolve/rev/name/train.parquet",
            },
        )
    ]

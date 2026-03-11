from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from datasets import Dataset
from datapreprocessor import api as ops
import datapreprocessor.load.download as load_module


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _make_out_path() -> Path:
    root = Path(__file__).resolve().parents[2] / "tests" / ".test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{uuid4().hex}.jsonl"


def test_download_adds_ids_by_default(monkeypatch):
    ds = Dataset.from_list(
        [
            {"translation": {"de": "eins", "en": "one"}},
            {"translation": {"de": "zwei", "en": "two"}},
        ]
    )
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    ops.download(dataset="dummy", config="de-en", split="train", output=out)

    rows = _read_jsonl(out)
    assert [row["id"] for row in rows] == [0, 1]


def test_download_can_disable_ids(monkeypatch):
    ds = Dataset.from_list([{"translation": {"de": "eins", "en": "one"}}])
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    ops.download(
        dataset="dummy", config="de-en", split="train", output=out, include_ids=False
    )

    rows = _read_jsonl(out)
    assert "id" not in rows[0]


def test_download_raises_if_id_exists_and_overwrite_disabled(monkeypatch):
    ds = Dataset.from_list(
        [{"id": 42, "translation": {"de": "eins", "en": "one"}}]
    )
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    try:
        ops.download(dataset="dummy", config="de-en", split="train", output=out)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "already exists" in str(exc)


def test_download_can_overwrite_existing_id(monkeypatch):
    ds = Dataset.from_list(
        [
            {"id": 42, "translation": {"de": "eins", "en": "one"}},
            {"id": 43, "translation": {"de": "zwei", "en": "two"}},
        ]
    )
    monkeypatch.setattr(load_module, "load_dataset", lambda *args, **kwargs: ds)
    out = _make_out_path()

    ops.download(
        dataset="dummy",
        config="de-en",
        split="train",
        output=out,
        overwrite_ids=True,
        start_id=100,
    )

    rows = _read_jsonl(out)
    assert [row["id"] for row in rows] == [100, 101]

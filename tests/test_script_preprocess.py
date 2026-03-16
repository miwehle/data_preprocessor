from __future__ import annotations

import runpy
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml


def test_script_preprocess_loads_yaml_and_calls_api(monkeypatch):
    tmp_path = Path(__file__).resolve().parent / ".test_artifacts" / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "Helsinki-NLP/europarl",
                "config": "de-en",
                "split": "train",
                "write_jsonl": False,
                "map_cfg": {"include_text": True},
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict] = []

    def fake_preprocess(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(sys, "argv", ["preprocess.py", str(config_path)])
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    import datapreprocessor.api as api

    monkeypatch.setattr(api, "preprocess", fake_preprocess)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"),
            run_name="__main__",
        )

    assert excinfo.value.code == 0
    assert calls == [
        {
            "dataset": "Helsinki-NLP/europarl",
            "config": "de-en",
            "split": "train",
            "download_cfg": None,
            "norm_cfg": None,
            "filter_cfg": None,
            "tokenize_cfg": None,
            "map_cfg": {"include_text": True},
            "write_jsonl": False,
        }
    ]

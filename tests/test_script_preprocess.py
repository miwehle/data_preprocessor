from __future__ import annotations

import runpy
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml


def test_script_preprocess_loads_yaml_and_calls_api(monkeypatch):
    tmp_path = Path(__file__).resolve().parents[1] / ".local_tmp" / "tests" / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "artifacts_dir": "/content/drive/MyDrive/nmt_lab/artifacts",
                "staging_dir": "/content/data_preprocessor_staging",
                "write_jsonl": False,
                "load_config": {"path_name": "Helsinki-NLP/europarl", "name": "de-en", "split": "train"},
                "tokenize_config": {"tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en"},
                "map_config": {"src_lang": "de", "tgt_lang": "en", "include_text": True},
                "split_config": {"split_ratio": {"train": 0.9, "val": 0.1}, "seed": 17},
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict] = []

    def fake_preprocess(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(sys, "argv", ["preprocess.py", str(config_path)])
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    import data_preprocessor as api

    monkeypatch.setattr(api, "preprocess", fake_preprocess)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"), run_name="__main__"
        )

    assert excinfo.value.code == 0
    assert calls == [
        {
            "load_config": api.LoadConfig(
                path_name="Helsinki-NLP/europarl", name="de-en", split="train"
            ),
            "norm_config": None,
            "filter_config": None,
            "tokenize_config": api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
            "map_config": api.MapConfig(src_lang="de", tgt_lang="en", include_text=True),
            "split_config": api.SplitConfig(split_ratio={"train": 0.9, "val": 0.1}, seed=17),
            "artifacts_dir": "/content/drive/MyDrive/nmt_lab/artifacts",
            "staging_dir": "/content/data_preprocessor_staging",
            "write_jsonl": False,
        }
    ]

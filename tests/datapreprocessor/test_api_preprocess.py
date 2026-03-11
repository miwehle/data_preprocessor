from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from datapreprocessor import api as ops


def _run_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / "tests" / ".test_artifacts"
    run_dir = root / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _patch_common_io(monkeypatch, *, capture_save: bool, calls: list[tuple[str, dict]]) -> None:
    monkeypatch.setattr(ops, "load", lambda path: [])
    if capture_save:
        monkeypatch.setattr(
            ops,
            "save",
            lambda examples, output_path: calls.append(("save", {"output_path": output_path})),
        )
    else:
        monkeypatch.setattr(ops, "save", lambda examples, output_path: None)


def _patch_stage_spies(monkeypatch, calls: list[tuple[str, dict]]) -> None:
    def _record(name: str):
        return lambda **kwargs: calls.append((name, kwargs))

    monkeypatch.setattr(ops, "download", _record("download"))
    monkeypatch.setattr(ops, "norm", _record("norm"))
    monkeypatch.setattr(ops, "filter", _record("filter"))
    monkeypatch.setattr(ops, "tokenize", _record("tokenize"))
    monkeypatch.setattr(ops, "map", _record("map"))


def _patch_training_token_ids(monkeypatch) -> None:
    monkeypatch.setattr(ops, "create_hf_tokenizer", lambda model_name: object())
    monkeypatch.setattr(
        ops,
        "resolve_training_token_ids",
        lambda tokenizer: {
            "src_pad_id": 58100,
            "tgt_pad_id": 58100,
            "tgt_bos_id": 58101,
            "tgt_eos_id": 0,
        },
    )


def test_ops_preprocess_calls_stages_in_order(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.chdir(_run_dir())
    _patch_common_io(monkeypatch, capture_save=True, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)

    ops.preprocess(
        download_cfg={"max_records": 123},
        map_cfg={"include_text": True},
    )

    assert [name for name, _ in calls] == ["download", "norm", "filter", "tokenize", "map", "save"]
    assert calls[0][1]["max_records"] == 123
    assert calls[0][1]["include_ids"] is True
    assert calls[-2][1]["include_text"] is True


def test_ops_preprocess_accepts_path_overrides(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.chdir(_run_dir())
    _patch_common_io(monkeypatch, capture_save=False, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)

    ops.preprocess(paths={"map_output": "C:/custom/final.jsonl"})

    assert calls[-1][1]["output_path"] == Path("C:/custom/final.jsonl")


def test_ops_preprocess_derives_filesystem_dataset_name(monkeypatch):
    seen_raw_output_paths: list[Path] = []
    seen_map_output_paths: list[Path] = []
    monkeypatch.chdir(_run_dir())

    def fake_download(**kwargs):
        seen_raw_output_paths.append(Path(kwargs["output"]))

    def fake_map(**kwargs):
        seen_map_output_paths.append(Path(kwargs["output_path"]))

    _patch_common_io(monkeypatch, capture_save=False, calls=[])
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(ops, "download", fake_download)
    monkeypatch.setattr(ops, "norm", lambda **kwargs: None)
    monkeypatch.setattr(ops, "filter", lambda **kwargs: None)
    monkeypatch.setattr(ops, "tokenize", lambda **kwargs: None)
    monkeypatch.setattr(ops, "map", fake_map)

    ops.preprocess(dataset="Org/My-Data Set+V1")

    assert seen_raw_output_paths
    assert seen_map_output_paths
    assert seen_raw_output_paths[0].name == "My-Data_Set_V1.raw.jsonl"
    assert seen_map_output_paths[0].name == "My-Data_Set_V1.mapped.jsonl"


def test_ops_preprocess_passes_training_token_ids_to_map(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.chdir(_run_dir())

    _patch_common_io(monkeypatch, capture_save=False, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)

    ops.preprocess()

    map_call = next(kwargs for name, kwargs in calls if name == "map")
    assert map_call["tgt_bos_id"] == 58101
    assert map_call["tgt_eos_id"] == 0


def test_ops_preprocess_writes_dataset_meta(monkeypatch):
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    _patch_common_io(monkeypatch, capture_save=False, calls=[])
    _patch_stage_spies(monkeypatch, [])
    _patch_training_token_ids(monkeypatch)

    ops.preprocess()

    meta_path = run_dir / "europarl_de-en_train" / "dataset_meta.json"
    assert meta_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta == {
        "schema_version": 1,
        "tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en",
        "src_lang": "de",
        "tgt_lang": "en",
        "id_field": "id",
        "src_field": "src_ids",
        "tgt_field": "tgt_ids",
        "tgt_bos_id": 58101,
        "tgt_eos_id": 0,
    }

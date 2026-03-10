from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))
from ipy import ops


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


def test_ops_preprocess_calls_stages_in_order(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.chdir(_run_dir())
    _patch_common_io(monkeypatch, capture_save=True, calls=calls)
    _patch_stage_spies(monkeypatch, calls)

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

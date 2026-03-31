from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import yaml

from data_preprocessor import api as ops


def _run_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / ".local_tmp" / "tests"
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
    class FakeTokenizer:
        vocab_size = 58102

    monkeypatch.setattr(ops, "create_hf_tokenizer", lambda model_name: FakeTokenizer())
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


def test_preprocess_calls_stages_in_order(monkeypatch):
    calls: list[tuple[str, dict]] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)
    _patch_common_io(monkeypatch, capture_save=True, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(ops, "_artifacts_root", lambda: run_dir / "artifacts" / "datasets")

    ops.preprocess(
        download_cfg={"dataset": "Helsinki-NLP/europarl", "config": "de-en", "split": "train", "max_examples": 123},
        norm_cfg={"changes": ["strip_edges", "collapse_whitespace"]},
        tokenize_cfg={
            "tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en",
            "max_seq_len": 256,
        },
        map_cfg={"src_lang": "de", "tgt_lang": "en", "include_text": True},
    )

    assert [name for name, _ in calls] == ["download", "norm", "filter", "tokenize", "map", "save"]
    assert calls[0][1]["max_examples"] == 123
    assert "include_ids" not in calls[0][1]
    norm_call = next(kwargs for name, kwargs in calls if name == "norm")
    assert norm_call["norm_cfg"] == {"changes": ["strip_edges", "collapse_whitespace"]}
    tokenize_call = next(kwargs for name, kwargs in calls if name == "tokenize")
    assert tokenize_call["max_seq_len"] == 256
    assert tokenize_call["src_lang"] == "de"
    assert calls[-2][1]["include_text"] is True
    assert calls[0][1]["output"] == (
        run_dir
        / "artifacts"
        / "datasets"
        / "europarl_de-en_train_123_staging"
        / "europarl_raw.jsonl"
    )


def test_preprocess_derives_filesystem_dataset_name(monkeypatch):
    seen_raw_output_paths: list[Path] = []
    seen_map_output_paths: list[Path] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    def fake_download(**kwargs):
        seen_raw_output_paths.append(Path(kwargs["output"]))

    def fake_map(**kwargs):
        seen_map_output_paths.append(Path(kwargs["output_path"]))

    _patch_common_io(monkeypatch, capture_save=False, calls=[])
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(ops, "_artifacts_root", lambda: run_dir / "artifacts" / "datasets")
    monkeypatch.setattr(ops, "download", fake_download)
    monkeypatch.setattr(ops, "norm", lambda **kwargs: None)
    monkeypatch.setattr(ops, "filter", lambda **kwargs: None)
    monkeypatch.setattr(ops, "tokenize", lambda **kwargs: None)
    monkeypatch.setattr(ops, "map", fake_map)

    ops.preprocess(
        download_cfg={"dataset": "Org/My-Data Set+V1", "config": "de-en", "split": "train"},
        tokenize_cfg={"tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en"},
        map_cfg={"src_lang": "de", "tgt_lang": "en"},
    )

    assert seen_raw_output_paths
    assert seen_map_output_paths
    assert seen_raw_output_paths[0].name == "My-Data_Set_V1_raw.jsonl"
    assert seen_map_output_paths[0].name == "My-Data_Set_V1_mapped.jsonl"
    assert seen_raw_output_paths[0].parent.name == "My-Data_Set_V1_de-en_train_staging"
    assert seen_raw_output_paths[0].parent.parent.name == "datasets"
    assert seen_map_output_paths[0].parent.name == "My-Data_Set_V1_de-en_train_staging"


def test_preprocess_passes_training_token_ids_to_map(monkeypatch):
    calls: list[tuple[str, dict]] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    _patch_common_io(monkeypatch, capture_save=False, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(ops, "_artifacts_root", lambda: run_dir / "artifacts" / "datasets")

    ops.preprocess(
        download_cfg={"dataset": "Helsinki-NLP/europarl", "config": "de-en", "split": "train"},
        tokenize_cfg={
            "tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en",
            "max_seq_len": 256,
        },
        map_cfg={"src_lang": "de", "tgt_lang": "en"},
    )

    map_call = next(kwargs for name, kwargs in calls if name == "map")
    assert map_call["tgt_bos_id"] == 58101
    assert map_call["tgt_eos_id"] == 0


def test_filter_uses_configured_predicates(monkeypatch):
    seen = {}

    monkeypatch.setattr(ops, "load", lambda path: [{"translation": {"de": "x", "en": "y"}}])
    monkeypatch.setattr(ops, "save", lambda examples, output_path: list(examples))

    def fake_filter_examples(ds, keep_fn):
        seen["text_flaws"] = keep_fn.keywords["text_flaws"]
        seen["pair_flaws"] = keep_fn.keywords["pair_flaws"]
        return ds

    monkeypatch.setattr(ops, "filter_examples", fake_filter_examples)

    ops.filter(
        input_path="in.jsonl",
        output_path="out.jsonl",
        flaw_report_path=None,
        filter_cfg={
            "predicates": ["is_blank", ["is_too_short", {"min": 5}]],
            "pair_predicates": ["are_equal"],
        },
    )

    assert [f.__name__ for f in seen["text_flaws"][:1]] == ["is_blank"]
    assert seen["text_flaws"][1].func.__name__ == "is_too_short"
    assert seen["text_flaws"][1].keywords == {"min": 5}
    assert [f.__name__ for f in seen["pair_flaws"]] == ["are_equal"]


def test_preprocess_writes_dataset_manifest(monkeypatch):
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    monkeypatch.setattr(ops, "load", lambda path: [{"id": 1}, {"id": 2}, {"id": 3}])
    monkeypatch.setattr(ops, "save", lambda examples, output_path: None)
    _patch_stage_spies(monkeypatch, [])
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(ops, "_artifacts_root", lambda: run_dir / "artifacts" / "datasets")

    ops.preprocess(
        download_cfg={"dataset": "Helsinki-NLP/europarl", "config": "de-en", "split": "train"},
        tokenize_cfg={
            "tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en",
            "max_seq_len": 256,
        },
        map_cfg={"src_lang": "de", "tgt_lang": "en"},
    )

    manifest_path = (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train" / "dataset_manifest.yaml")
    assert manifest_path.is_file()

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en",
        "src_lang": "de",
        "tgt_lang": "en",
        "id_field": "id",
        "src_field": "src_ids",
        "tgt_field": "tgt_ids",
        "base_vocab_size": 58102,
        "src_vocab_size": 58102,
        "tgt_vocab_size": 58102,
        "src_pad_id": 58100,
        "tgt_pad_id": 58100,
        "tgt_bos_id": 58101,
        "tgt_eos_id": 0,
        "num_examples": 3,
        "configured_max_seq_len": 256,
    }


def test_preprocess_uses_incremented_dataset_dir(monkeypatch):
    calls: list[tuple[str, dict]] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)
    _patch_common_io(monkeypatch, capture_save=True, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(ops, "_artifacts_root", lambda: run_dir / "artifacts" / "datasets")

    (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train"
    ).mkdir(
        parents=True, exist_ok=True
    )

    ops.preprocess(
        download_cfg={"dataset": "Helsinki-NLP/europarl", "config": "de-en", "split": "train"},
        tokenize_cfg={"tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en"},
        map_cfg={"src_lang": "de", "tgt_lang": "en"},
    )

    assert (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train (1)"
    ).is_dir()
    assert calls[0][1]["output"] == (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train_staging (1)" / "europarl_raw.jsonl"
    )
    assert calls[-1][1]["output_path"] == (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train (1)"
    )


def test_preprocess_logs_to_dataset_preprocess_log(monkeypatch):
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    monkeypatch.setattr(ops, "download_examples", lambda **kwargs: [])
    monkeypatch.setattr(ops, "load", lambda path: [])
    monkeypatch.setattr(ops, "save", lambda examples, output_path: None)
    monkeypatch.setattr(ops, "norm_examples", lambda ds, changes, norm_reporter: ds)
    monkeypatch.setattr(ops, "filter_examples", lambda ds, keep_fn: ds)
    monkeypatch.setattr(ops, "map_examples", lambda ds, **kwargs: ds)
    monkeypatch.setattr(
        ops,
        "tokenize_examples",
        lambda ds, tokenizer, tokenize_reporter=None, max_seq_len=None, src_lang="de", **kwargs: iter(ds),
    )
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(ops, "_artifacts_root", lambda: run_dir / "artifacts" / "datasets")

    ops.preprocess(
        download_cfg={"dataset": "Helsinki-NLP/europarl", "config": "de-en", "split": "train"},
        tokenize_cfg={"tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en"},
        map_cfg={"src_lang": "de", "tgt_lang": "en"},
    )

    log_path = run_dir / "artifacts" / "datasets" / "europarl_de-en_train" / "preprocess.log"
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "Start download" in log_text
    assert "Finished map" in log_text

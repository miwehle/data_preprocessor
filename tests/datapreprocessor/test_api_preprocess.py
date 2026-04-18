from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import yaml

from data_preprocessor import api


def _run_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / ".local_tmp" / "tests"
    run_dir = root / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _patch_common_io(monkeypatch, *, capture_save: bool, calls: list[tuple[str, dict]]) -> None:
    monkeypatch.setattr(api.io, "load", lambda path: [])
    if capture_save:
        monkeypatch.setattr(
            api.io, "save", lambda examples, output_path: calls.append(("save", {"output_path": output_path}))
        )
    else:
        monkeypatch.setattr(api.io, "save", lambda examples, output_path: None)


def _patch_stage_spies(monkeypatch, calls: list[tuple[str, dict]]) -> None:
    def _record(name: str):
        return lambda *args: calls.append((name, {"args": args}))

    monkeypatch.setattr(api, "load", _record("load"))
    monkeypatch.setattr(api, "norm", _record("norm"))
    monkeypatch.setattr(api, "filter", _record("filter"))
    monkeypatch.setattr(api, "tokenize", _record("tokenize"))
    monkeypatch.setattr(api, "map", _record("map"))


def _patch_training_token_ids(monkeypatch) -> None:
    class FakeTokenizer:
        vocab_size = 58102

    monkeypatch.setattr(api, "create_hf_tokenizer", lambda model_name: FakeTokenizer())
    monkeypatch.setattr(
        api,
        "resolve_training_token_ids",
        lambda tokenizer: {"src_pad_id": 58100, "tgt_pad_id": 58100, "tgt_bos_id": 58101, "tgt_eos_id": 0},
    )


def test_preprocess_calls_stages_in_order(monkeypatch):
    calls: list[tuple[str, dict]] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)
    _patch_common_io(monkeypatch, capture_save=True, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")

    api.preprocess(
        load_cfg=api.LoadConfig(
            path_name="Helsinki-NLP/europarl", name="de-en", split="train", max_examples=123
        ),
        norm_cfg=api.NormConfig(changes=["strip_edges", "collapse_whitespace"]),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en", max_seq_len=256),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en", include_text=True),
    )

    assert [name for name, _ in calls] == ["load", "norm", "filter", "tokenize", "map", "save"]
    assert calls[0][1]["args"][0].max_examples == 123
    assert calls[0][1]["args"][0].include_ids is True
    norm_call = next(kwargs for name, kwargs in calls if name == "norm")
    assert norm_call["args"][0].changes == ["strip_edges", "collapse_whitespace"]
    tokenize_call = next(kwargs for name, kwargs in calls if name == "tokenize")
    assert tokenize_call["args"][0].max_seq_len == 256
    assert tokenize_call["args"][0].src_lang == "de"
    assert calls[-2][1]["args"][0].include_text is True
    assert calls[0][1]["args"][1] == (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train_123_staging" / "europarl_raw.jsonl"
    )


def test_preprocess_derives_filesystem_dataset_name(monkeypatch):
    seen_raw_output_paths: list[Path] = []
    seen_map_output_paths: list[Path] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    def fake_load(config, output_path):
        seen_raw_output_paths.append(Path(output_path))

    def fake_map(config, input_path, output_path):
        seen_map_output_paths.append(Path(output_path))

    _patch_common_io(monkeypatch, capture_save=False, calls=[])
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")
    monkeypatch.setattr(api, "load", fake_load)
    monkeypatch.setattr(api, "norm", lambda config, input_path, output_path, report_path=None: None)
    monkeypatch.setattr(api, "filter", lambda config, input_path, output_path, report_path=None: None)
    monkeypatch.setattr(api, "tokenize", lambda config, input_path, output_path, report_path=None: None)
    monkeypatch.setattr(api, "map", fake_map)

    api.preprocess(
        load_cfg=api.LoadConfig(path_name="Org/My-Data Set+V1", name="de-en", split="train"),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
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
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")

    api.preprocess(
        load_cfg=api.LoadConfig(path_name="Helsinki-NLP/europarl", name="de-en", split="train"),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en", max_seq_len=256),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
    )

    map_call = next(kwargs for name, kwargs in calls if name == "map")
    assert map_call["args"][0].tgt_bos_id == 58101
    assert map_call["args"][0].tgt_eos_id == 0


def test_filter_uses_configured_predicates(monkeypatch):
    seen = {}

    monkeypatch.setattr(api.io, "load", lambda path: [{"translation": {"de": "x", "en": "y"}}])
    monkeypatch.setattr(api.io, "save", lambda examples, output_path: list(examples))

    def fake_filter_examples(ds, keep_fn):
        seen["text_flaws"] = keep_fn.keywords["text_flaws"]
        seen["pair_flaws"] = keep_fn.keywords["pair_flaws"]
        return ds

    monkeypatch.setattr(api, "filter_examples", fake_filter_examples)

    api.filter(
        api.FilterConfig(
            predicates=["is_blank", ["is_too_short", {"min": 5}]], pair_predicates=["are_equal"]
        ),
        "in.jsonl",
        "out.jsonl",
        None,
    )

    assert [f.__name__ for f in seen["text_flaws"][:1]] == ["is_blank"]
    assert seen["text_flaws"][1].func.__name__ == "is_too_short"
    assert seen["text_flaws"][1].keywords == {"min": 5}
    assert [f.__name__ for f in seen["pair_flaws"]] == ["are_equal"]


def test_preprocess_writes_dataset_manifest(monkeypatch):
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    monkeypatch.setattr(api.io, "load", lambda path: [{"id": 1}, {"id": 2}, {"id": 3}])
    monkeypatch.setattr(api.io, "save", lambda examples, output_path: None)
    _patch_stage_spies(monkeypatch, [])
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")

    api.preprocess(
        load_cfg=api.LoadConfig(path_name="Helsinki-NLP/europarl", name="de-en", split="train"),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en", max_seq_len=256),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
    )

    manifest_path = run_dir / "artifacts" / "datasets" / "europarl_de-en_train" / "dataset_manifest.yaml"
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
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")

    (run_dir / "artifacts" / "datasets" / "europarl_de-en_train").mkdir(parents=True, exist_ok=True)

    api.preprocess(
        load_cfg=api.LoadConfig(path_name="Helsinki-NLP/europarl", name="de-en", split="train"),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
    )

    assert (run_dir / "artifacts" / "datasets" / "europarl_de-en_train (1)").is_dir()
    assert calls[0][1]["args"][1] == (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train (1)_staging" / "europarl_raw.jsonl"
    )
    assert calls[-1][1]["output_path"] == (run_dir / "artifacts" / "datasets" / "europarl_de-en_train (1)")


def test_preprocess_logs_to_dataset_preprocess_log(monkeypatch):
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    monkeypatch.setattr(api, "load_examples", lambda config: [])
    monkeypatch.setattr(api.io, "load", lambda path: [])
    monkeypatch.setattr(api.io, "save", lambda examples, output_path: None)
    monkeypatch.setattr(api, "norm_examples", lambda ds, config=None, norm_reporter=None: ds)
    monkeypatch.setattr(api, "filter_examples", lambda ds, keep_fn: ds)
    monkeypatch.setattr(api, "map_examples", lambda ds, config: ds)
    monkeypatch.setattr(
        api, "tokenize_examples", lambda ds, config, tokenizer, tokenize_reporter=None: iter(ds)
    )
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")

    api.preprocess(
        load_cfg=api.LoadConfig(path_name="Helsinki-NLP/europarl", name="de-en", split="train"),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
    )

    log_path = run_dir / "artifacts" / "datasets" / "europarl_de-en_train" / "preprocess.log"
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "Start load" in log_text
    assert "Finished map" in log_text


def test_preprocess_uses_separate_staging_dir(monkeypatch):
    calls: list[tuple[str, dict]] = []
    run_dir = _run_dir()
    staging_root = run_dir / "colab-staging"
    _patch_common_io(monkeypatch, capture_save=True, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")

    api.preprocess(
        load_cfg=api.LoadConfig(path_name="Helsinki-NLP/europarl", name="de-en", split="train"),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
        staging_dir=staging_root,
    )

    assert calls[0][1]["args"][1] == staging_root / "europarl_de-en_train_staging" / "europarl_raw.jsonl"
    assert calls[-1][1]["output_path"] == run_dir / "artifacts" / "datasets" / "europarl_de-en_train"


def test_preprocess_uses_dataset_name_for_generic_downloads(monkeypatch):
    seen_raw_output_paths: list[Path] = []
    seen_map_output_paths: list[Path] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)

    def fake_load(config, output_path):
        seen_raw_output_paths.append(Path(output_path))

    def fake_map(config, input_path, output_path):
        seen_map_output_paths.append(Path(output_path))

    _patch_common_io(monkeypatch, capture_save=False, calls=[])
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")
    monkeypatch.setattr(api, "load", fake_load)
    monkeypatch.setattr(api, "norm", lambda config, input_path, output_path, report_path=None: None)
    monkeypatch.setattr(api, "filter", lambda config, input_path, output_path, report_path=None: None)
    monkeypatch.setattr(api, "tokenize", lambda config, input_path, output_path, report_path=None: None)
    monkeypatch.setattr(api, "map", fake_map)

    api.preprocess(
        load_cfg=api.LoadConfig(
            path_name="parquet",
            split="train",
            data_files="https://huggingface.co/datasets/org/ds/resolve/rev/name/train.parquet",
            dataset_name="iwslt2017-de-en",
        ),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
    )

    assert seen_raw_output_paths[0].name == "iwslt2017-de-en_raw.jsonl"
    assert seen_map_output_paths[0].name == "iwslt2017-de-en_mapped.jsonl"
    assert seen_raw_output_paths[0].parent.name == "iwslt2017-de-en_train_staging"
    assert seen_map_output_paths[0].parent.name == "iwslt2017-de-en_train_staging"


def test_preprocess_requires_dataset_name_for_generic_downloads(monkeypatch):
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)
    _patch_common_io(monkeypatch, capture_save=False, calls=[])
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")

    try:
        api.preprocess(
            load_cfg=api.LoadConfig(
                path_name="parquet",
                split="train",
                data_files="https://huggingface.co/datasets/org/ds/resolve/rev/name/train.parquet",
            ),
            tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
            map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "dataset_name" in str(exc)


def test_preprocess_calls_split_after_writing_base_dataset(monkeypatch):
    calls: list[tuple[str, dict]] = []
    run_dir = _run_dir()
    monkeypatch.chdir(run_dir)
    _patch_common_io(monkeypatch, capture_save=True, calls=calls)
    _patch_stage_spies(monkeypatch, calls)
    _patch_training_token_ids(monkeypatch)
    monkeypatch.setattr(api, "_artifacts_root", lambda: run_dir / "artifacts")
    monkeypatch.setattr(api, "split", lambda config: calls.append(("split", {"config": config})))

    api.preprocess(
        load_cfg=api.LoadConfig(path_name="Helsinki-NLP/europarl", name="de-en", split="train"),
        tokenize_cfg=api.TokenizeConfig(tokenizer_model_name="Helsinki-NLP/opus-mt-de-en"),
        map_cfg=api.MapConfig(src_lang="de", tgt_lang="en"),
        split_cfg=api.SplitConfig(split_ratio={"train": 0.9, "val": 0.1}, seed=13),
    )

    assert [name for name, _ in calls] == ["load", "norm", "filter", "tokenize", "map", "save", "split"]
    split_call = calls[-1][1]["config"]
    assert split_call == api.SplitConfig(
        dataset=str(run_dir / "artifacts" / "datasets" / "europarl_de-en_train"),
        split_ratio={"train": 0.9, "val": 0.1},
        seed=13,
    )
    config_text = (
        run_dir / "artifacts" / "datasets" / "europarl_de-en_train" / "preprocess_config.yaml"
    ).read_text(encoding="utf-8")
    assert "split_cfg:" in config_text
    assert "seed: 13" in config_text

"""Thin public orchestration layer."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, nullcontext
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from lab_infrastructure.logging import get_logger, log_calls
from lab_infrastructure.run_config import write_run_config

from data_preprocessor.filter import FlawReport, filter_examples, keep, pair_predicates, predicates
from data_preprocessor.load import load_examples
from data_preprocessor.map import map_examples
from data_preprocessor.norm import NormReport, changes as norm_changes, norm_examples
from data_preprocessor.shared import (
    Example,
    FilterConfig,
    LoadConfig,
    MapConfig,
    NormConfig,
    SplitConfig,
    TokenizeConfig,
)
from data_preprocessor.split import split_dataset
from data_preprocessor.tokenizer import (
    TokenizeReport,
    create_hf_tokenizer,
    resolve_training_token_ids,
    tokenize_examples,
)

from . import io


def _dataset_name_for_filesystem(dataset: str) -> str:
    base = dataset.rsplit("/", maxsplit=1)[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "dataset"


def _run_dirs(run_name: str, final_root: Path, staging_root: Path | None) -> tuple[Path, Path | None]:
    staging_dir = None if staging_root is None else staging_root / f"{run_name}_staging"
    return final_root / run_name, staging_dir


def _next_available_run_name(base_name: str, final_root: Path, staging_root: Path | None) -> str:
    run_name = base_name
    i = 1
    while True:
        output_dir, staging_dir = _run_dirs(run_name, final_root, staging_root)
        if not output_dir.exists() and (staging_dir is None or not staging_dir.exists()):
            return run_name
        run_name = f"{base_name} ({i})"
        i += 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts_root() -> Path:
    return _repo_root().parent / "artifacts"


def _datasets_root(artifacts_dir: str | Path | None) -> Path:
    return (Path(artifacts_dir) if artifacts_dir is not None else _artifacts_root()) / "datasets"


def _staging_root(artifacts_dir: str | Path | None, staging_dir: str | Path | None) -> Path:
    return Path(staging_dir) if staging_dir is not None else _datasets_root(artifacts_dir)


def _stage_name(config: object) -> str:
    return type(config).__name__.removesuffix("Config").lower()


def _report_context(config: object, staging_dir: Path | None):
    if staging_dir is None:
        return nullcontext(None)
    report_path = staging_dir / f"{_stage_name(config)}_report.txt"
    if isinstance(config, NormConfig):
        return closing(NormReport.from_path(report_path, debug=config.norm_debug))
    if isinstance(config, FilterConfig):
        return closing(FlawReport.from_path(report_path))
    if isinstance(config, TokenizeConfig):
        return closing(TokenizeReport.from_path(report_path, debug=config.tokenize_debug))
    return nullcontext(None)


def _validate_preprocess_configs(
    load_config: LoadConfig,
    tokenize_config: TokenizeConfig,
    map_config: MapConfig,
    training_token_ids: dict[str, int],
) -> None:
    if load_config.data_files is not None and load_config.dataset_name is None:
        raise ValueError("load_config.dataset_name is required when load_config.data_files is set.")
    if tokenize_config.src_lang is not None and tokenize_config.src_lang != map_config.src_lang:
        raise ValueError(
            f"Conflicting src_lang values: tokenize_config={tokenize_config.src_lang!r}, "
            f"map_config={map_config.src_lang!r}."
        )
    if map_config.id_key is not None and map_config.id_key != load_config.id_field:
        raise ValueError(
            f"Conflicting id field values: load_config.id_field={load_config.id_field!r}, "
            f"map_config={map_config.id_key!r}."
        )
    if map_config.tgt_bos_id is not None and map_config.tgt_bos_id != training_token_ids["tgt_bos_id"]:
        raise ValueError(
            f"Conflicting tgt_bos_id values: map_config={map_config.tgt_bos_id!r}, "
            f"tokenizer={training_token_ids['tgt_bos_id']!r}."
        )
    if map_config.tgt_eos_id is not None and map_config.tgt_eos_id != training_token_ids["tgt_eos_id"]:
        raise ValueError(
            f"Conflicting tgt_eos_id values: map_config={map_config.tgt_eos_id!r}, "
            f"tokenizer={training_token_ids['tgt_eos_id']!r}."
        )


def _collect_dataset_metadata(
    tokenizer: Any,
    training_token_ids: dict[str, int],
    tokenize_config: TokenizeConfig,
    map_config: MapConfig,
    num_examples: int,
) -> dict[str, object]:
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if tokenizer_vocab_size is None:
        raise ValueError("Tokenizer must define vocab_size for dataset metadata.")
    base_vocab_size = int(tokenizer_vocab_size)
    return {
        "schema_version": 1,
        "tokenizer_model_name": tokenize_config.tokenizer_model_name,
        "src_lang": map_config.src_lang,
        "tgt_lang": map_config.tgt_lang,
        "id_field": map_config.id_key or "id",
        "src_field": "src_ids",
        "tgt_field": "tgt_ids",
        "base_vocab_size": base_vocab_size,
        "src_vocab_size": max(base_vocab_size, training_token_ids["src_pad_id"] + 1),
        "tgt_vocab_size": max(
            base_vocab_size,
            training_token_ids["tgt_pad_id"] + 1,
            training_token_ids["tgt_bos_id"] + 1,
            training_token_ids["tgt_eos_id"] + 1,
        ),
        "src_pad_id": training_token_ids["src_pad_id"],
        "tgt_pad_id": training_token_ids["tgt_pad_id"],
        "tgt_bos_id": training_token_ids["tgt_bos_id"],
        "tgt_eos_id": training_token_ids["tgt_eos_id"],
        "num_examples": num_examples,
        "configured_max_seq_len": tokenize_config.max_seq_len,
    }


_log_calls = log_calls(get_logger("data_preprocessor", log_path=_artifacts_root() / "data_preprocessor.log"))


@_log_calls
def load(config: LoadConfig) -> Iterable[Example]:
    return load_examples(config)


@_log_calls
def norm(ds: Iterable[Example], config: NormConfig, report: Any | None = None) -> Iterator[Example]:
    resolved_config = NormConfig(changes=norm_changes(config.changes) or (), norm_debug=config.norm_debug)
    yield from norm_examples(ds, resolved_config, report)


@_log_calls
def filter(ds: Iterable[Example], config: FilterConfig, report: Any | None = None) -> Iterator[Example]:
    ps = predicates(config.predicates) or ()
    pps = pair_predicates(config.pair_predicates) or ()
    yield from filter_examples(ds, partial(keep, flaw_reporter=report, text_flaws=ps, pair_flaws=pps))


@_log_calls
def tokenize(
    ds: Iterable[Example],
    config: TokenizeConfig,
    tokenizer: Any | None = None,
    report: Any | None = None,
) -> Iterator[Example]:
    if config.max_seq_len is not None and config.max_seq_len < 0:
        raise ValueError("max_seq_len must be >= 0 or None.")
    yield from tokenize_examples(ds, config, tokenizer or create_hf_tokenizer(config.tokenizer_model_name), report)


@_log_calls
def map(ds: Iterable[Example], config: MapConfig, report: Any | None = None) -> Iterator[Example]:
    yield from map_examples(ds, config)


@_log_calls
def split(config: SplitConfig) -> None:
    split_dataset(config)


@_log_calls
def preprocess(
    load_config: LoadConfig,
    tokenize_config: TokenizeConfig,
    map_config: MapConfig,
    *,
    norm_config: NormConfig | None = None,
    filter_config: FilterConfig | None = None,
    split_config: SplitConfig | None = None,
    artifacts_dir: str | Path | None = None,
    staging_dir: str | Path | None = None,
    write_snapshots: bool = False,
) -> None:
    """Run the full preprocessing pipeline and produce translation-training output.

    If the configured tokenizer does not define a target BOS token (for example
    Marian/OPUS-MT), preprocessing materializes one for the mapped training
    output.

    This does not yet create batched tensors. A later collation step (usually in
    the training data loader) still pads src/tgt sequences to the batch-local max
    length, stacks them into tensors, and returns the batch IDs alongside them.
    """
    def snapshot(ds: Iterable[Example], config: object) -> Iterable[Example]:
        if resolved_staging_dir is None:
            return ds
        path = resolved_staging_dir / f"{dataset_name}_{_stage_name(config)}.jsonl"
        io.save(ds, path)
        return io.load(path)

    def run_stage(
        stage: Callable[..., Iterable[Example]], ds: Iterable[Example], config: object, *args: Any
    ) -> Iterable[Example]:
        with _report_context(config, resolved_staging_dir) as report:
            return snapshot(stage(ds, config, *args, report), config)

    # initialize configs and output paths
    norm_config = norm_config or NormConfig()
    filter_config = filter_config or FilterConfig()
    dataset_name = load_config.dataset_name or _dataset_name_for_filesystem(load_config.path_name)
    dataset_dir_name = dataset_name
    if load_config.dataset_name is None and load_config.name is not None:
        dataset_dir_name = f"{dataset_dir_name}_{load_config.name}"
    dataset_dir_name = f"{dataset_dir_name}_{load_config.split}"
    if load_config.max_examples is not None:
        dataset_dir_name = f"{dataset_dir_name}_{load_config.max_examples}"
    final_root = _datasets_root(artifacts_dir)
    resolved_staging_root = _staging_root(artifacts_dir, staging_dir) if write_snapshots else None
    run_name = _next_available_run_name(dataset_dir_name, final_root, resolved_staging_root)
    preprocessed_output, resolved_staging_dir = _run_dirs(run_name, final_root, resolved_staging_root)
    preprocessed_output.mkdir(parents=True, exist_ok=True)
    if resolved_staging_dir is not None:
        resolved_staging_dir.mkdir(parents=True, exist_ok=True)
    get_logger("data_preprocessor", log_path=preprocessed_output / "preprocess.log")

    tokenizer = create_hf_tokenizer(tokenize_config.tokenizer_model_name)
    training_token_ids = resolve_training_token_ids(tokenizer)
    _validate_preprocess_configs(load_config, tokenize_config, map_config, training_token_ids)

    # fill missing config fields
    resolved_tokenize_config = replace(tokenize_config, src_lang=tokenize_config.src_lang or map_config.src_lang)
    resolved_map_config = replace(
        map_config,
        id_key=map_config.id_key or load_config.id_field,
        tgt_bos_id=training_token_ids["tgt_bos_id"] if map_config.tgt_bos_id is None else map_config.tgt_bos_id,
        tgt_eos_id=training_token_ids["tgt_eos_id"] if map_config.tgt_eos_id is None else map_config.tgt_eos_id,
    )
    resolved_split_config = (
        replace(split_config, dataset=str(preprocessed_output)) if split_config is not None else None
    )

    # write preprocess_config.yaml
    write_run_config(
        preprocessed_output / "preprocess_config.yaml",
        {
            "dataset_schema_version": "1",
            "write_snapshots": write_snapshots,
            "artifacts_dir": None if artifacts_dir is None else str(artifacts_dir),
            "staging_dir": None if staging_dir is None else str(staging_dir),
            "load_config": asdict(load_config),
            "norm_config": asdict(norm_config),
            "filter_config": asdict(filter_config),
            "tokenize_config": asdict(resolved_tokenize_config),
            "map_config": asdict(resolved_map_config),
            "split_config": None if resolved_split_config is None else asdict(resolved_split_config),
        },
        repo_root=_repo_root(),
        git_key_prefix="data_preprocessor",
    )

    # core
    ds = snapshot(load(load_config), load_config)
    ds = run_stage(norm, ds, norm_config)
    ds = run_stage(filter, ds, filter_config)
    ds = run_stage(tokenize, ds, resolved_tokenize_config, tokenizer)
    mapped = list(run_stage(map, ds, resolved_map_config))

    # write dataset_manifest.yaml
    dataset_manifest = _collect_dataset_metadata(
        tokenizer, training_token_ids, resolved_tokenize_config, resolved_map_config, len(mapped)
    )
    with (preprocessed_output / "dataset_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_manifest, f, sort_keys=False, allow_unicode=True)

    io.save(mapped, preprocessed_output)
    if resolved_split_config is not None:
        split(resolved_split_config)

"""Thin public orchestration layer.

This module coordinates I/O and delegates transformation logic to the
specialized `data_preprocessor.*` modules. Keep business logic out of this file.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
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
from data_preprocessor.shared import FilterConfig, LoadConfig, MapConfig, NormConfig, SplitConfig, TokenizeConfig
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
    if safe:
        return safe
    return "dataset"


def _run_output_roots(
    run_name: str, staging_root: Path, final_root: Path, run_index: int | None = None
) -> tuple[Path, Path]:
    suffix = f" ({run_index})" if run_index is not None else ""
    return (staging_root / f"{run_name}_staging{suffix}", final_root / f"{run_name}{suffix}")


def _next_available_run_name(base_name: str, staging_root: Path, final_root: Path) -> str:
    staging_dir, final_dir = _run_output_roots(base_name, staging_root, final_root)
    if not staging_dir.exists() and not final_dir.exists():
        return base_name

    i = 1
    while True:
        candidate = f"{base_name} ({i})"
        staging_dir, final_dir = _run_output_roots(base_name, staging_root, final_root, i)
        if not staging_dir.exists() and not final_dir.exists():
            return candidate
        i += 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts_root() -> Path:
    return _repo_root().parent / "artifacts"


def _datasets_root(artifacts_dir: str | Path | None) -> Path:
    return (Path(artifacts_dir) if artifacts_dir is not None else _artifacts_root()) / "datasets"


def _staging_root(artifacts_dir: str | Path | None, staging_dir: str | Path | None) -> Path:
    return Path(staging_dir) if staging_dir is not None else _datasets_root(artifacts_dir)


def _default_paths(
    *, run_name: str, dataset_name: str, write_jsonl: bool, staging_root: Path, final_root: Path
) -> dict[str, Path]:
    staging_dir, preprocessed_dir = _run_output_roots(run_name, staging_root, final_root)
    suffix = ".jsonl" if write_jsonl else ""
    return {
        "raw_output": staging_dir / f"{dataset_name}_raw{suffix}",
        "norm_output": staging_dir / f"{dataset_name}_norm{suffix}",
        "filter_output": staging_dir / f"{dataset_name}_filtered{suffix}",
        "tokenize_output": staging_dir / f"{dataset_name}_tokenized{suffix}",
        "map_output": staging_dir / f"{dataset_name}_mapped{suffix}",
        "preprocessed_output": preprocessed_dir,
        "norm_report": staging_dir / "norm_report.txt",
        "flaw_report": staging_dir / "flaw_report.txt",
        "tokenize_report": staging_dir / "tokenize_report.txt",
        "preprocess_config": preprocessed_dir / "preprocess_config.yaml",
        "dataset_manifest": preprocessed_dir / "dataset_manifest.yaml",
    }


def _run_with_optional_report(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None,
    make_report: Callable[[str | Path], Any],
    transform: Callable[[Iterable[dict], Any | None], Iterable[dict]],
) -> None:
    ds = io.load(input_path)
    report_context = closing(make_report(report_path)) if report_path is not None else nullcontext(None)
    with report_context as report:
        io.save(transform(ds, report), output_path)


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
            f"map_config.id_key={map_config.id_key!r}."
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
def load(config: LoadConfig, output_path: str | Path) -> None:
    io.save(load_examples(config), output_path)


@_log_calls
def norm(
    config: NormConfig,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = "norm_report.txt",
) -> None:
    _run_with_optional_report(
        input_path,
        output_path,
        report_path,
        lambda path: NormReport.from_path(path, debug=config.norm_debug),
        lambda ds, report: norm_examples(
            ds, NormConfig(changes=norm_changes(config.changes) or (), norm_debug=config.norm_debug), report
        ),
    )


@_log_calls
def filter(
    config: FilterConfig,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = "flaw_report.txt",
) -> None:
    ps = predicates(config.predicates) or ()
    pps = pair_predicates(config.pair_predicates) or ()
    _run_with_optional_report(
        input_path,
        output_path,
        report_path,
        FlawReport.from_path,
        lambda ds, report: filter_examples(
            ds, partial(keep, flaw_reporter=report, text_flaws=ps, pair_flaws=pps)
        ),
    )


@_log_calls
def tokenize(
    config: TokenizeConfig,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = "tokenize_report.txt",
) -> None:
    tokenizer = create_hf_tokenizer(config.tokenizer_model_name)
    if config.max_seq_len is not None and config.max_seq_len < 0:
        raise ValueError("max_seq_len must be >= 0 or None.")
    _run_with_optional_report(
        input_path,
        output_path,
        report_path,
        lambda path: TokenizeReport.from_path(path, debug=config.tokenize_debug),
        lambda ds, report: tokenize_examples(ds, config, tokenizer, report),
    )


@_log_calls
def map(config: MapConfig, input_path: str | Path, output_path: str | Path) -> None:
    io.save(map_examples(io.load(input_path), config), output_path)


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
    write_jsonl: bool = True,
) -> None:
    """Run the full preprocessing pipeline and produce translation-training output.

    If the configured tokenizer does not define a target BOS token (for example
    Marian/OPUS-MT), preprocessing materializes one for the mapped training
    output.

    This does not yet create batched tensors. A later collation step (usually in
    the training data loader) still pads src/tgt sequences to the batch-local max
    length, stacks them into tensors, and returns the batch IDs alongside them.
    """

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
    resolved_staging_root = _staging_root(artifacts_dir, staging_dir)
    run_name = _next_available_run_name(dataset_dir_name, resolved_staging_root, final_root)
    paths = _default_paths(
        run_name=run_name,
        dataset_name=dataset_name,
        write_jsonl=write_jsonl,
        staging_root=resolved_staging_root,
        final_root=final_root,
    )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    get_logger("data_preprocessor", log_path=paths["preprocessed_output"] / "preprocess.log")

    tokenizer = create_hf_tokenizer(tokenize_config.tokenizer_model_name)
    training_token_ids = resolve_training_token_ids(tokenizer)

    _validate_preprocess_configs(load_config, tokenize_config, map_config, training_token_ids)

    # fill missing config fields
    resolved_tokenize_config = replace(
        tokenize_config, src_lang=tokenize_config.src_lang or map_config.src_lang
    )
    resolved_map_config = replace(
        map_config,
        id_key=map_config.id_key or load_config.id_field,
        tgt_bos_id=training_token_ids["tgt_bos_id"] if map_config.tgt_bos_id is None else map_config.tgt_bos_id,
        tgt_eos_id=training_token_ids["tgt_eos_id"] if map_config.tgt_eos_id is None else map_config.tgt_eos_id,
    )
    resolved_split_config = (
        replace(split_config, dataset=str(paths["preprocessed_output"])) if split_config is not None else None
    )

    # write preprocess_config.yaml
    write_run_config(
        paths["preprocess_config"],
        {
            "dataset_schema_version": "1",
            "write_jsonl": write_jsonl,
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
    load(load_config, paths["raw_output"])
    norm(norm_config, paths["raw_output"], paths["norm_output"], paths["norm_report"])
    filter(filter_config, paths["norm_output"], paths["filter_output"], paths["flaw_report"])
    tokenize(
        resolved_tokenize_config,
        paths["filter_output"],
        paths["tokenize_output"],
        paths["tokenize_report"],
    )
    map(resolved_map_config, paths["tokenize_output"], paths["map_output"])

    mapped = io.load(paths["map_output"])

    # write dataset_manifest.yaml
    dataset_manifest = _collect_dataset_metadata(
        tokenizer, training_token_ids, resolved_tokenize_config, resolved_map_config, len(mapped)
    )
    with paths["dataset_manifest"].open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_manifest, f, sort_keys=False, allow_unicode=True)

    io.save(mapped, paths["preprocessed_output"])
    if resolved_split_config is not None:
        split(resolved_split_config)


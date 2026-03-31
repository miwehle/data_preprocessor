"""Thin public orchestration layer.

This module coordinates I/O and delegates transformation logic to the
specialized `data_preprocessor.*` modules. Keep business logic out of this file.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterable
from contextlib import closing, nullcontext
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import yaml

from data_preprocessor.filter import FlawReport, filter_examples, keep, pair_predicates, predicates
from data_preprocessor.load import download_examples
from data_preprocessor.map import map_examples
from data_preprocessor.metadata import build_dataset_meta
from data_preprocessor.norm import NormReport, changes as norm_changes, norm_examples
from data_preprocessor.shared import configure_data_preprocessor_logging, log_calls
from data_preprocessor.tokenizer import (
    TokenizeReport,
    create_hf_tokenizer,
    resolve_training_token_ids,
    tokenize_examples,
)

from .io import load, save


_log_calls = log_calls(lambda: _artifacts_root().parent / "data_preprocessor.log")


def _dataset_name_for_filesystem(dataset: str) -> str:
    base = dataset.rsplit("/", maxsplit=1)[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if safe:
        return safe
    return "dataset"


def _run_output_roots(run_root: Path, run_index: int | None = None) -> tuple[Path, Path]:
    suffix = f" ({run_index})" if run_index is not None else ""
    return (
        run_root.with_name(f"{run_root.name}_staging{suffix}"),
        run_root.with_name(f"{run_root.name}{suffix}"),
    )


def _next_available_run_dir(base_dir: Path) -> Path:
    staging_dir, final_dir = _run_output_roots(base_dir)
    if not staging_dir.exists() and not final_dir.exists():
        return base_dir

    i = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name} ({i})")
        staging_dir, final_dir = _run_output_roots(base_dir, i)
        if not staging_dir.exists() and not final_dir.exists():
            return candidate
        i += 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts_root() -> Path:
    return _repo_root().parent / "artifacts" / "datasets"


def _current_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(_repo_root()),
        )
        commit = out.strip()
        return commit if commit else None
    except Exception:
        return None


def _current_git_status() -> str:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(_repo_root()),
        )
        return "no local changes" if out.strip() == "" else "local changes exist"
    except Exception:
        return "local changes exist"


def _default_paths(*, dataset_dir: Path, dataset_name: str, write_jsonl: bool) -> dict[str, Path]:
    run_index = None
    match = re.search(r" \((\d+)\)$", dataset_dir.name)
    if match:
        run_index = int(match.group(1))
        dataset_dir = dataset_dir.with_name(dataset_dir.name[: match.start()])

    staging_dir, preprocessed_dir = _run_output_roots(dataset_dir, run_index)
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
    *,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None,
    make_report: Callable[[str | Path], Any],
    transform: Callable[[Iterable[dict], Any | None], Iterable[dict]],
) -> None:
    ds = load(input_path)
    report_context = (
        closing(make_report(report_path)) if report_path is not None else nullcontext(None)
    )
    with report_context as report:
        save(transform(ds, report), output_path)


@_log_calls
def download(
    *,
    dataset: str,
    config: str,
    split: str,
    source_format: Literal["hf", "hf_parquet"] = "hf",
    revision: str = "main",
    parquet_basename: str | None = None,
    output: str | Path,
    max_examples: int | None = None,
    include_ids: bool = True,
    id_field: str = "id",
    start_id: int = 0,
    overwrite_ids: bool = False,
) -> None:
    """Download one dataset split and optionally assign stable example IDs.

    See the ``download_examples`` docstring for parameter details.
    """
    examples = download_examples(
        dataset=dataset,
        config=config,
        split=split,
        source_format=source_format,
        revision=revision,
        parquet_basename=parquet_basename,
        max_examples=max_examples,
        include_ids=include_ids,
        id_field=id_field,
        start_id=start_id,
        overwrite_ids=overwrite_ids,
    )
    save(examples, output)


@_log_calls
def norm(
    *,
    input_path: str | Path,
    output_path: str | Path,
    norm_report_path: str | Path | None = "norm_report.txt",
    norm_debug: bool = False,
    norm_cfg: dict[str, Any] | None = None,
) -> None:
    """Normalize text examples and optionally write a norm report."""
    norm_cfg = norm_cfg or {}
    changes = norm_changes(norm_cfg.get("changes")) or ()

    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=norm_report_path,
        make_report=lambda path: NormReport.from_path(path, debug=norm_debug),
        transform=lambda ds, report: norm_examples(ds, changes=changes, norm_reporter=report),
    )


@_log_calls
def filter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    flaw_report_path: str | Path | None = "flaw_report.txt",
    filter_cfg: dict[str, Any] | None = None,
) -> None:
    """Filter examples and optionally write a flaw report."""
    filter_cfg = filter_cfg or {}
    ps = predicates(filter_cfg.get("predicates")) or ()
    pps = pair_predicates(filter_cfg.get("pair_predicates")) or ()

    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=flaw_report_path,
        make_report=FlawReport.from_path,
        transform=lambda ds, report: filter_examples(
            ds,
            partial(keep, flaw_reporter=report, text_flaws=ps, pair_flaws=pps),
        ),
    )


@_log_calls
def tokenize(
    *,
    input_path: str | Path,
    output_path: str | Path,
    tokenize_report_path: str | Path | None = "tokenize_report.txt",
    tokenizer_model_name: str,
    tokenizer_kwargs: dict | None = None,
    tokenize_debug: bool = False,
    max_seq_len: int | None = None,
    src_lang: str = "de",
) -> None:
    """Tokenize both translation sides and write nested tokenized_translation output.

    If ``max_seq_len`` is set, examples whose untruncated source or target token
    list is longer than that limit are dropped entirely before tokenized output
    is written.
    """
    tokenizer = create_hf_tokenizer(tokenizer_model_name)
    if max_seq_len is not None and max_seq_len < 0:
        raise ValueError("max_seq_len must be >= 0 or None.")

    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=tokenize_report_path,
        make_report=lambda path: TokenizeReport.from_path(path, debug=tokenize_debug),
        transform=lambda ds, report: tokenize_examples(
            ds,
            tokenizer=tokenizer,
            tokenize_reporter=report,
            max_seq_len=max_seq_len,
            src_lang=src_lang,
            tokenizer_kwargs=tokenizer_kwargs,
        ),
    )


@_log_calls
def map(
    *,
    input_path: str | Path,
    output_path: str | Path,
    id_key: str = "id",
    tokenized_key: str = "tokenized_translation",
    src_lang: str,
    tgt_lang: str,
    tgt_bos_id: int | None = None,
    tgt_eos_id: int | None = None,
    include_text: bool = False,
) -> None:
    """Map tokenized examples to training fields and normalize target boundary tokens if needed."""
    ds = load(input_path)
    mapped = map_examples(
        ds,
        id_key=id_key,
        tokenized_key=tokenized_key,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        tgt_bos_id=tgt_bos_id,
        tgt_eos_id=tgt_eos_id,
        include_text=include_text,
    )
    save(mapped, output_path)


@_log_calls
def preprocess(
    *,
    download_cfg: dict[str, Any] | None = None,
    norm_cfg: dict[str, Any] | None = None,
    filter_cfg: dict[str, Any] | None = None,
    tokenize_cfg: dict[str, Any] | None = None,
    map_cfg: dict[str, Any] | None = None,
    write_jsonl: bool = True,
) -> None:
    """Run the full preprocessing pipeline and produce translation-training output.

    The `*_cfg` dictionaries are passed through to the stage functions in this
    module and must include those functions' required parameters.

    If the configured tokenizer does not define a target BOS token (for example
    Marian/OPUS-MT), preprocessing materializes one for the mapped training
    output.

    This does not yet create batched tensors. A later collation step (usually in
    the training data loader) still pads src/tgt sequences to the batch-local max
    length, stacks them into tensors, and returns the batch IDs alongside them.
    """
    download_cfg = download_cfg or {}
    norm_cfg = norm_cfg or {}
    filter_cfg = filter_cfg or {}
    tokenize_cfg = tokenize_cfg or {}
    map_cfg = map_cfg or {}

    dataset_name = _dataset_name_for_filesystem(download_cfg["dataset"])

    resolved_download_cfg = {
        "max_examples": None,
        "include_ids": True,
        "id_field": "id",
        "start_id": 0,
        "overwrite_ids": False,
        **download_cfg,
    }
    resolved_tokenize_cfg = {
        "tokenizer_kwargs": None,
        "tokenize_debug": False,
        "max_seq_len": None,
        "src_lang": map_cfg.get("src_lang", "de"),
        **tokenize_cfg,
    }
    tokenizer = create_hf_tokenizer(resolved_tokenize_cfg["tokenizer_model_name"])
    training_token_ids = resolve_training_token_ids(tokenizer)
    resolved_map_cfg = {
        "id_key": resolved_download_cfg["id_field"],
        "tokenized_key": "tokenized_translation",
        "tgt_bos_id": training_token_ids["tgt_bos_id"],
        "tgt_eos_id": training_token_ids["tgt_eos_id"],
        "include_text": False,
        **map_cfg,
    }

    dataset_dir_name = (
        f"{dataset_name}_{resolved_download_cfg['config']}_{resolved_download_cfg['split']}"
    )
    max_examples = resolved_download_cfg["max_examples"]
    if max_examples is not None:
        dataset_dir_name = f"{dataset_dir_name}_{max_examples}"

    dataset_dir = _next_available_run_dir(_artifacts_root() / dataset_dir_name)
    paths = _default_paths(dataset_dir=dataset_dir, dataset_name=dataset_name, write_jsonl=write_jsonl)

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    configure_data_preprocessor_logging(log_path=paths["preprocessed_output"] / "preprocess.log")

    parameters = {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "data_preprocessor_git_commit": _current_git_commit(),
        "data_preprocessor_git_status": _current_git_status(),
        "dataset_schema_version": "1",
        "write_jsonl": write_jsonl,
        "download_cfg": resolved_download_cfg,
        "norm_cfg": norm_cfg,
        "filter_cfg": filter_cfg,
        "tokenize_cfg": resolved_tokenize_cfg,
        "map_cfg": resolved_map_cfg,
    }
    with paths["preprocess_config"].open("w", encoding="utf-8") as f:
        yaml.safe_dump(parameters, f, sort_keys=False, allow_unicode=True)

    stages: list[tuple[Callable[..., None], dict[str, Any]]] = [
        (
            download,
            {
                "output": paths["raw_output"],
                **download_cfg,
            },
        ),
        (
            norm,
            {
                "input_path": paths["raw_output"],
                "output_path": paths["norm_output"],
                "norm_report_path": paths["norm_report"],
                "norm_debug": norm_cfg.get("norm_debug", False),
                "norm_cfg": norm_cfg,
            },
        ),
        (
            filter,
            {
                "input_path": paths["norm_output"],
                "output_path": paths["filter_output"],
                "flaw_report_path": paths["flaw_report"],
                "filter_cfg": filter_cfg,
            },
        ),
        (
            tokenize,
            {
                "input_path": paths["filter_output"],
                "output_path": paths["tokenize_output"],
                "tokenize_report_path": paths["tokenize_report"],
                **resolved_tokenize_cfg,
            },
        ),
        (
            map,
            {
                "input_path": paths["tokenize_output"],
                "output_path": paths["map_output"],
                **map_cfg,
                "tgt_bos_id": map_cfg.get("tgt_bos_id", training_token_ids["tgt_bos_id"]),
                "tgt_eos_id": map_cfg.get("tgt_eos_id", training_token_ids["tgt_eos_id"]),
            },
        ),
    ]
    for stage_fn, stage_kwargs in stages:
        stage_fn(**stage_kwargs)

    mapped = load(paths["map_output"])
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if tokenizer_vocab_size is None:
        raise ValueError("Tokenizer must define vocab_size for dataset metadata.")
    base_vocab_size = int(tokenizer_vocab_size)
    src_vocab_size = max(base_vocab_size, training_token_ids["src_pad_id"] + 1)
    tgt_vocab_size = max(
        base_vocab_size,
        training_token_ids["tgt_pad_id"] + 1,
        training_token_ids["tgt_bos_id"] + 1,
        training_token_ids["tgt_eos_id"] + 1,
    )
    dataset_manifest = build_dataset_meta(
        tokenizer_model_name=resolved_tokenize_cfg["tokenizer_model_name"],
        src_lang=resolved_map_cfg["src_lang"],
        tgt_lang=resolved_map_cfg["tgt_lang"],
        id_field=resolved_map_cfg["id_key"],
        src_field="src_ids",
        tgt_field="tgt_ids",
        base_vocab_size=base_vocab_size,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_id=training_token_ids["src_pad_id"],
        tgt_pad_id=training_token_ids["tgt_pad_id"],
        tgt_bos_id=training_token_ids["tgt_bos_id"],
        tgt_eos_id=training_token_ids["tgt_eos_id"],
        num_examples=len(mapped),
        configured_max_seq_len=resolved_tokenize_cfg["max_seq_len"],
    )
    with paths["dataset_manifest"].open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_manifest, f, sort_keys=False, allow_unicode=True)
    save(mapped, paths["preprocessed_output"])

"""Thin public orchestration layer.

This module coordinates I/O and delegates transformation logic to the
specialized `datapreprocessor.*` modules. Keep business logic out of this file.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterable
from contextlib import closing, nullcontext
from datetime import UTC, datetime
from functools import partial, wraps
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from datapreprocessor.filter import FlawReport, filter_examples, keep
from datapreprocessor.load import download_examples
from datapreprocessor.map import map_examples
from datapreprocessor.metadata import build_dataset_meta
from datapreprocessor.norm import NormReport, norm_examples
from datapreprocessor.tokenizer import (
    TokenizeReport,
    create_hf_tokenizer,
    resolve_training_token_ids,
    tokenize_examples,
)

from .io import load, save


def _log_calls(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            path = _artifacts_root().parent / "data_preprocessor.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            handler.formatter.default_msec_format = "%s,%03d"
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        logger.info("Start %s", func.__name__)
        started = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("Finished %s in %.3fs", func.__name__, perf_counter() - started)

    return wrapper


def _dataset_name_for_filesystem(dataset: str) -> str:
    base = dataset.rsplit("/", maxsplit=1)[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if safe:
        return safe
    return "dataset"


def _next_available_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    i = 1
    while True:
        candidate = Path(f"{base_dir} ({i})")
        if not candidate.exists():
            return candidate
        i += 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts_root() -> Path:
    return _repo_root().parent / "artifacts" / "datasets"


def _current_git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
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
    if write_jsonl:
        raw_output = dataset_dir / "raw" / f"{dataset_name}.raw.jsonl"
        norm_output = dataset_dir / "interim" / f"{dataset_name}.norm.jsonl"
        filter_output = dataset_dir / "interim" / f"{dataset_name}.filtered.jsonl"
        tokenize_output = dataset_dir / "interim" / f"{dataset_name}.tokenized.jsonl"
        map_output = dataset_dir / "interim" / f"{dataset_name}.mapped.jsonl"
    else:
        raw_output = dataset_dir / "raw" / f"{dataset_name}.raw"
        norm_output = dataset_dir / "interim" / f"{dataset_name}.norm"
        filter_output = dataset_dir / "interim" / f"{dataset_name}.filtered"
        tokenize_output = dataset_dir / "interim" / f"{dataset_name}.tokenized"
        map_output = dataset_dir / "interim" / f"{dataset_name}.mapped"

    return {
        "raw_output": raw_output,
        "norm_output": norm_output,
        "filter_output": filter_output,
        "tokenize_output": tokenize_output,
        "map_output": map_output,
        "preprocessed_output": dataset_dir / "preprocessed" / f"{dataset_name}.preprocessed",
        "norm_report": dataset_dir / "interim" / "norm_report.txt",
        "flaw_report": dataset_dir / "interim" / "flaw_report.txt",
        "tokenize_report": dataset_dir / "interim" / "tokenize_report.txt",
        "preprocess_config": dataset_dir / "preprocessed" / "preprocess_config.yaml",
        "dataset_manifest": dataset_dir / "preprocessed" / "dataset_manifest.yaml",
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

    print(f"Wrote {output_path}")
    if report_path is not None:
        print(f"Wrote {report_path}")


@_log_calls
def download(
    *,
    dataset: str,
    config: str,
    split: str,
    output: str | Path,
    max_records: int | None = None,
    include_ids: bool = True,
    id_field: str = "id",
    start_id: int = 0,
    overwrite_ids: bool = False,
) -> None:
    """Download one dataset split and optionally assign stable example IDs."""
    records = download_examples(
        dataset=dataset,
        config=config,
        split=split,
        max_records=max_records,
        include_ids=include_ids,
        id_field=id_field,
        start_id=start_id,
        overwrite_ids=overwrite_ids,
    )
    save(records, output)
    print(f"Wrote {output}")


@_log_calls
def norm(
    *,
    input_path: str | Path,
    output_path: str | Path,
    norm_report_path: str | Path | None = "norm_report.txt",
    norm_debug: bool = False,
) -> None:
    """Normalize text examples and optionally write a norm report."""
    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=norm_report_path,
        make_report=lambda path: NormReport.from_path(path, debug=norm_debug),
        transform=lambda ds, report: norm_examples(ds, norm_reporter=report),
    )


@_log_calls
def filter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    flaw_report_path: str | Path | None = "flaw_report.txt",
) -> None:
    """Filter examples and optionally write a flaw report."""
    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=flaw_report_path,
        make_report=FlawReport.from_path,
        transform=lambda ds, report: filter_examples(ds, partial(keep, flaw_reporter=report)),
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
) -> None:
    """Tokenize both translation sides and write nested tokenized_translation output."""
    tokenizer = create_hf_tokenizer(tokenizer_model_name)

    effective_kwargs = {"truncation": True, "max_length": 256, **(tokenizer_kwargs or {})}

    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=tokenize_report_path,
        make_report=lambda path: TokenizeReport.from_path(path, debug=tokenize_debug),
        transform=lambda ds, report: tokenize_examples(
            ds, tokenizer=tokenizer, tokenize_reporter=report, tokenizer_kwargs=effective_kwargs
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
    print(f"Wrote {output_path}")


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
        "max_records": None,
        "include_ids": True,
        "id_field": "id",
        "start_id": 0,
        "overwrite_ids": False,
        **download_cfg,
    }
    resolved_tokenize_cfg = {
        "tokenizer_kwargs": None,
        "tokenize_debug": False,
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

    dataset_dir_name = dataset_name
    max_records = resolved_download_cfg["max_records"]
    if max_records is not None:
        dataset_dir_name = f"{dataset_dir_name}_{max_records}"

    dataset_dir = _next_available_run_dir(_artifacts_root() / dataset_dir_name)
    paths = _default_paths(dataset_dir=dataset_dir, dataset_name=dataset_name, write_jsonl=write_jsonl)

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    parameters = {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "datapreprocessor_git_commit": _current_git_commit_short(),
        "datapreprocessor_git_status": _current_git_status(),
        "dataset_schema_version": "1",
        "write_jsonl": write_jsonl,
        "paths": {key: str(value) for key, value in paths.items()},
        "download_cfg": resolved_download_cfg,
        "norm_cfg": norm_cfg,
        "filter_cfg": filter_cfg,
        "tokenize_cfg": resolved_tokenize_cfg,
        "map_cfg": resolved_map_cfg,
    }
    with paths["preprocess_config"].open("w", encoding="utf-8") as f:
        yaml.safe_dump(parameters, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {paths['preprocess_config']}")

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
                **norm_cfg,
            },
        ),
        (
            filter,
            {
                "input_path": paths["norm_output"],
                "output_path": paths["filter_output"],
                "flaw_report_path": paths["flaw_report"],
                **filter_cfg,
            },
        ),
        (
            tokenize,
            {
                "input_path": paths["filter_output"],
                "output_path": paths["tokenize_output"],
                "tokenize_report_path": paths["tokenize_report"],
                **tokenize_cfg,
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
    )
    with paths["dataset_manifest"].open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_manifest, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {paths['dataset_manifest']}")
    save(mapped, paths["preprocessed_output"])
    print(f"Wrote {paths['preprocessed_output']}")

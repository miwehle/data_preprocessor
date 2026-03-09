"""Thin IPython-facing orchestration layer.

This module coordinates I/O and delegates transformation logic to
`src/datapreprocessor/*`. Keep business logic out of this file.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable
from contextlib import closing, nullcontext
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from datapreprocessor.filter import FlawReport, filter_examples, keep
from datapreprocessor.load import download_records
from datapreprocessor.map import to_training_schema
from datapreprocessor.norm import NormReport, norm_examples
from datapreprocessor.tokenizer import (
    TokenizeReport,
    create_hf_tokenizer,
    tokenize_examples,
)

from .io import load, save


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


def _run_with_optional_report(
    *,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None,
    make_report: Callable[[str | Path], Any],
    transform: Callable[[Iterable[dict], Any | None], Iterable[dict]],
) -> None:
    ds = load(input_path)
    report_context = closing(make_report(report_path)) if report_path is not None else nullcontext(None)
    with report_context as report:
        save(transform(ds, report), output_path)

    print(f"Wrote {output_path}")
    if report_path is not None:
        print(f"Wrote {report_path}")


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
    records = download_records(
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


def norm(
    *,
    input_path: str | Path,
    output_path: str | Path,
    norm_report_path: str | Path | None = "norm_report.txt",
    norm_debug: bool = False,
) -> None:
    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=norm_report_path,
        make_report=lambda path: NormReport.from_path(path, debug=norm_debug),
        transform=lambda ds, report: norm_examples(ds, norm_reporter=report),
    )


def filter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    flaw_report_path: str | Path | None = "flaw_report.txt",
) -> None:
    _run_with_optional_report(
        input_path=input_path,
        output_path=output_path,
        report_path=flaw_report_path,
        make_report=FlawReport.from_path,
        transform=lambda ds, report: filter_examples(ds, partial(keep, flaw_reporter=report)),
    )


def tokenize(
    *,
    input_path: str | Path,
    output_path: str | Path,
    tokenize_report_path: str | Path | None = "tokenize_report.txt",
    tokenizer_model_name: str = "Helsinki-NLP/opus-mt-de-en",
    tokenizer_kwargs: dict | None = None,
    tokenize_debug: bool = False,
) -> None:
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


def map(
    *,
    input_path: str | Path,
    output_path: str | Path,
    id_key: str = "id",
    tokenized_key: str = "tokenized_translation",
    src_lang: str = "de",
    tgt_lang: str = "en",
    include_text: bool = False,
) -> None:
    ds = load(input_path)
    mapped = to_training_schema(
        ds,
        id_key=id_key,
        tokenized_key=tokenized_key,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        include_text=include_text,
    )
    save(mapped, output_path)
    print(f"Wrote {output_path}")


def preprocess(
    *,
    dataset: str = "Helsinki-NLP/europarl",
    config: str = "de-en",
    split: str = "train",
    paths: dict[str, str | Path] | None = None,
    download_cfg: dict[str, Any] | None = None,
    norm_cfg: dict[str, Any] | None = None,
    filter_cfg: dict[str, Any] | None = None,
    tokenize_cfg: dict[str, Any] | None = None,
    map_cfg: dict[str, Any] | None = None,
) -> None:
    """Run the end-to-end preprocessing pipeline with practical defaults."""

    dataset_name = _dataset_name_for_filesystem(dataset)

    effective_download_cfg = {
        "max_records": None,
        "include_ids": True,
        "id_field": "id",
        "start_id": 0,
        "overwrite_ids": False,
        **(download_cfg or {}),
    }
    effective_norm_cfg = {"norm_debug": False, **(norm_cfg or {})}
    effective_filter_cfg = {**(filter_cfg or {})}
    effective_tokenize_cfg = {
        "tokenizer_model_name": "Helsinki-NLP/opus-mt-de-en",
        "tokenizer_kwargs": None,
        "tokenize_debug": False,
        **(tokenize_cfg or {}),
    }
    effective_map_cfg = {
        "id_key": effective_download_cfg["id_field"],
        "tokenized_key": "tokenized_translation",
        "src_lang": "de",
        "tgt_lang": "en",
        "include_text": False,
        **(map_cfg or {}),
    }

    run_dir_name = f"{dataset_name}_{config}_{split}"
    max_records = effective_download_cfg["max_records"]
    if max_records is not None:
        run_dir_name = f"{run_dir_name}_{max_records}"

    run_dir = _next_available_run_dir(Path.cwd() / run_dir_name)
    effective_paths: dict[str, Path] = {
        "raw_output": run_dir / f"{dataset_name}.raw.jsonl",
        "norm_output": run_dir / f"{dataset_name}.norm.jsonl",
        "filter_output": run_dir / f"{dataset_name}.filtered.jsonl",
        "tokenize_output": run_dir / f"{dataset_name}.tokenized.jsonl",
        "map_output": run_dir / f"{dataset_name}.mapped",
        "norm_report": run_dir / "norm_report.txt",
        "flaw_report": run_dir / "flaw_report.txt",
        "tokenize_report": run_dir / "tokenize_report.txt",
        "preprocess_config": run_dir / "preprocess_config.json",
    }
    if paths:
        for key, value in paths.items():
            if key in effective_paths:
                effective_paths[key] = Path(value)

    run_dir.mkdir(parents=True, exist_ok=True)

    parameters = {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "datapreprocessor_git_commit": _current_git_commit_short(),
        "datapreprocessor_git_status": _current_git_status(),
        "dataset_schema_version": "1",
        "dataset": dataset,
        "config": config,
        "split": split,
        "paths": {key: str(value) for key, value in effective_paths.items()},
        "download_cfg": effective_download_cfg,
        "norm_cfg": effective_norm_cfg,
        "filter_cfg": effective_filter_cfg,
        "tokenize_cfg": effective_tokenize_cfg,
        "map_cfg": effective_map_cfg,
    }
    with effective_paths["preprocess_config"].open("w", encoding="utf-8") as f:
        json.dump(parameters, f, ensure_ascii=False, indent=2)
    print(f"Wrote {effective_paths['preprocess_config']}")

    download(
        dataset=dataset,
        config=config,
        split=split,
        output=effective_paths["raw_output"],
        max_records=effective_download_cfg["max_records"],
        include_ids=effective_download_cfg["include_ids"],
        id_field=effective_download_cfg["id_field"],
        start_id=effective_download_cfg["start_id"],
        overwrite_ids=effective_download_cfg["overwrite_ids"],
    )
    norm(
        input_path=effective_paths["raw_output"],
        output_path=effective_paths["norm_output"],
        norm_report_path=effective_paths["norm_report"],
        norm_debug=effective_norm_cfg["norm_debug"],
    )
    filter(
        input_path=effective_paths["norm_output"],
        output_path=effective_paths["filter_output"],
        flaw_report_path=effective_paths["flaw_report"],
        **effective_filter_cfg,
    )
    tokenize(
        input_path=effective_paths["filter_output"],
        output_path=effective_paths["tokenize_output"],
        tokenize_report_path=effective_paths["tokenize_report"],
        tokenizer_model_name=effective_tokenize_cfg["tokenizer_model_name"],
        tokenizer_kwargs=effective_tokenize_cfg["tokenizer_kwargs"],
        tokenize_debug=effective_tokenize_cfg["tokenize_debug"],
    )
    map(
        input_path=effective_paths["tokenize_output"],
        output_path=effective_paths["map_output"],
        id_key=effective_map_cfg["id_key"],
        tokenized_key=effective_map_cfg["tokenized_key"],
        src_lang=effective_map_cfg["src_lang"],
        tgt_lang=effective_map_cfg["tgt_lang"],
        include_text=effective_map_cfg["include_text"],
    )

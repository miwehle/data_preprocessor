from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from datapreprocessor.filter import FlawReport, filter_examples, keep
from datapreprocessor.norm import NormReport, norm_examples
from datapreprocessor.tokenizer import TokenizeReport, tokenize_examples

from .io import load, save


def _run_with_optional_report(
    *,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None,
    make_report: Callable[[str | Path], Any],
    transform: Callable[[Iterable[dict], Any | None], Iterable[dict]],
) -> None:
    ds = load(input_path)
    report = make_report(report_path) if report_path is not None else None
    try:
        save(transform(ds, report), output_path)
    finally:
        if report is not None:
            report.close()

    print(f"Wrote {output_path}")
    if report_path is not None:
        print(f"Wrote {report_path}")


def download(
    *, dataset: str, config: str, split: str, output: str | Path, max_records: int | None = None
) -> None:
    ds = load_dataset(dataset, config, split=split)
    records = ds if max_records is None else ds.select(range(min(max_records, len(ds))))
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
    model_name: str = "Helsinki-NLP/opus-mt-de-en",
    tokenizer_kwargs: dict | None = None,
    tokenize_debug: bool = False,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

"""Thin IPython-facing orchestration layer.

This module coordinates I/O and delegates transformation logic to
`src/datapreprocessor/*`. Keep business logic out of this file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import closing, nullcontext
from functools import partial
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from datapreprocessor.filter import FlawReport, filter_examples, keep
from datapreprocessor.load import download_records
from datapreprocessor.map import to_training_schema
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

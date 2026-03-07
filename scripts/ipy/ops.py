from __future__ import annotations

from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from datapreprocessor.filter import FlawReport, filter_examples, keep
from datapreprocessor.norm import NormReport, norm_examples
from datapreprocessor.tokenizer import TokenizeReport, tokenize_examples

from .io import load, save


def download(
    *,
    dataset: str,
    config: str,
    split: str,
    output: str | Path,
    max_records: int | None = None,
) -> None:
    ds = load_dataset(dataset, config, split=split)
    records = ds if max_records is None else ds.select(range(min(max_records, len(ds))))
    save(records, output)
    print(f"Wrote {output}")


def norm(
    *,
    input_path: str | Path,
    output_path: str | Path,
    norm_report_path: str | Path,
    norm_debug: bool = False,
) -> None:
    ds = load(input_path)
    report = NormReport.from_path(norm_report_path, debug=norm_debug)
    try:
        save(norm_examples(ds, norm_reporter=report), output_path)
    finally:
        report.close()
    print(f"Wrote {output_path}")
    print(f"Wrote {norm_report_path}")


def filter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    flaw_report_path: str | Path,
) -> None:
    ds = load(input_path)
    report = FlawReport.from_path(flaw_report_path)
    try:
        save(filter_examples(ds, partial(keep, flaw_reporter=report)), output_path)
    finally:
        report.close()
    print(f"Wrote {output_path}")
    print(f"Wrote {flaw_report_path}")


def tokenize(
    *,
    input_path: str | Path,
    output_path: str | Path,
    tokenize_report_path: str | Path,
    model_name: str = "Helsinki-NLP/opus-mt-de-en",
    tokenizer_kwargs: dict | None = None,
    tokenize_debug: bool = False,
) -> None:
    ds = load(input_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    report = TokenizeReport.from_path(tokenize_report_path, debug=tokenize_debug)

    effective_kwargs = {
        "truncation": True,
        "max_length": 256,
        **(tokenizer_kwargs or {}),
    }

    try:
        save(
            tokenize_examples(
                ds,
                tokenizer=tokenizer,
                tokenize_reporter=report,
                tokenizer_kwargs=effective_kwargs,
            ),
            output_path,
        )
    finally:
        report.close()

    print(f"Wrote {output_path}")
    print(f"Wrote {tokenize_report_path}")

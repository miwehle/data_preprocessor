from __future__ import annotations

from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from datapreprocessor.filter.filter import filter_examples
from datapreprocessor.filter.keep import FlawReport, keep
from datapreprocessor.norm.norm import norm_examples
from datapreprocessor.norm.norm_example import NormReport
from datapreprocessor.tokenizer.tokenize_example import TokenizeReport
from datapreprocessor.tokenizer.tokenizer import tokenize_examples

from .io import load_jsonl, write_jsonl


def download(
    *,
    dataset: str,
    config: str,
    split: str,
    output: str | Path,
    max_rows: int | None = None,
) -> int:
    ds = load_dataset(dataset, config, split=split)
    rows = ds if max_rows is None else (ex for i, ex in enumerate(ds) if i < max_rows)
    write_jsonl(rows, output)
    print(f"Wrote {output}")
    return 0


def norm(
    *,
    input_path: str | Path,
    output_path: str | Path,
    norm_report_path: str | Path,
    norm_debug: bool = False,
) -> int:
    ds = load_jsonl(input_path)
    report = NormReport.from_path(norm_report_path, debug=norm_debug)
    try:
        write_jsonl(norm_examples(ds, norm_reporter=report), output_path)
    finally:
        report.close()
    print(f"Wrote {output_path}")
    print(f"Wrote {norm_report_path}")
    return 0


def filter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    flaw_report_path: str | Path,
) -> int:
    ds = load_jsonl(input_path)
    report = FlawReport.from_path(flaw_report_path)
    try:
        write_jsonl(filter_examples(ds, partial(keep, flaw_reporter=report)), output_path)
    finally:
        report.close()
    print(f"Wrote {output_path}")
    print(f"Wrote {flaw_report_path}")
    return 0


def tokenize(
    *,
    input_path: str | Path,
    output_path: str | Path,
    tokenize_report_path: str | Path,
    model_name: str = "Helsinki-NLP/opus-mt-de-en",
    tokenizer_kwargs: dict | None = None,
    tokenize_debug: bool = False,
) -> int:
    ds = load_jsonl(input_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    report = TokenizeReport.from_path(tokenize_report_path, debug=tokenize_debug)

    effective_kwargs = {
        "truncation": True,
        "max_length": 256,
        **(tokenizer_kwargs or {}),
    }

    try:
        write_jsonl(
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
    return 0

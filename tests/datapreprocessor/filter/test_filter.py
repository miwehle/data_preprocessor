from pathlib import Path
from functools import partial

from datasets import load_dataset

from datapreprocessor.filter import FlawReport, filter_examples, keep


def test_filter_with_keep():
    root_dir = Path(__file__).resolve().parents[3]
    data_file = root_dir / "tests" / "data" / "testdata_de_en_1000.jsonl"
    report = FlawReport.from_path(root_dir / "flaw_report.txt")
    ds = load_dataset("json", data_files=str(data_file), split="train")
    it = filter_examples(ds, partial(keep, flaw_reporter=report))

    try:
        for ex in it:
            pass
    finally:
        report.close()
        actual = root_dir / "flaw_report.txt"
        expected = root_dir / "tests" / "expected" / "filter" / "flaw_report.txt"
        assert actual.read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")


def test_filter_preserves_id_field():
    ds = [
        {"id": 10, "translation": {"de": "guten tag", "en": "good day"}},
        {"id": 11, "translation": {"de": "x", "en": "x"}},
    ]
    out = list(filter_examples(ds, lambda ex: True))

    assert [ex["id"] for ex in out] == [10, 11]

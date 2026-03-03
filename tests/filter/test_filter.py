from pathlib import Path

from datasets import load_dataset

from datapreprocessor.filter.filter import filter_examples
from datapreprocessor.filter.keep import FlawReport, keep


def test_filter_with_keep():
    root_dir = Path(__file__).resolve().parents[2]
    data_file = root_dir / "data" / "testdata_de_en_100.jsonl"
    ds = load_dataset("json", data_files=str(data_file), split="train")
    it = filter_examples(ds, keep)

    try:
        for ex in it:
            pass
    finally:
        report = FlawReport.singleton()
        report.out.flush()
        actual_log = root_dir / "flaw_report.txt"
        expected_log = root_dir / "tests" / "expected" / "filter" / "flaw_report.txt"
        assert actual_log.read_text(encoding="utf-8") == expected_log.read_text(encoding="utf-8")

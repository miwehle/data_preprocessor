from pathlib import Path

from datasets import load_dataset

from datapreprocessor.checks.filter import filtered_examples
from datapreprocessor.checks.keep import FlawReport, keep


def test_filtered_examples_with_keep():
    root_dir = Path(__file__).resolve().parents[2]
    data_file = root_dir / "data" / "testdata_de_en_100.jsonl"
    ds = load_dataset("json", data_files=str(data_file), split="train")
    it = filtered_examples(ds, keep)

    try:
        for ex in it:
            pass
    finally:
        report = FlawReport.singleton()
        report.out.flush()
        actual_log = root_dir / "flaw_report.log"
        expected_log = root_dir / "tests" / "expected" / "checks" / "flaw_report.log"
        assert actual_log.read_text(encoding="utf-8") == expected_log.read_text(encoding="utf-8")

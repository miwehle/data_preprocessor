from pathlib import Path

from datasets import load_dataset

from datapreprocessor.norm.norm import norm_examples
from datapreprocessor.norm.norm_example import NormReport


def test_norm_report_matches_expected():
    root_dir = Path(__file__).resolve().parents[2]
    data_file = root_dir / "tests" / "data" / "testdata_de_en_1000.jsonl"
    report = NormReport.from_path(root_dir / "norm_report.txt")
    ds = load_dataset("json", data_files=str(data_file), split="train")
    it = norm_examples(ds, norm_reporter=report)

    try:
        for ex in it:
            pass
    finally:
        report.close()
        actual_report = root_dir / "norm_report.txt"
        expected_report = root_dir / "tests" / "expected" / "norm" / "norm_report.txt"
        assert actual_report.read_text(encoding="utf-8") == expected_report.read_text(encoding="utf-8")

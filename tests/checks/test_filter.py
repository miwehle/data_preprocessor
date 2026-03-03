from pathlib import Path

from datasets import load_dataset

from datapreprocessor.checks.filter import filtered_examples
from datapreprocessor.checks.keep import keep


def test_filtered_examples_with_keep():
    data_file = Path(__file__).resolve().parents[2] / "data" / "testdata_de_en_100.jsonl"
    ds = load_dataset("json", data_files=str(data_file), split="train")
    it = filtered_examples(ds, keep)

    try:
        for ex in it:
            pass
    finally:
        pass
    pass

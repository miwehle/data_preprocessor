from pathlib import Path

from datasets import load_dataset

from datapreprocessor.tokenizer import TokenizeReport, tokenize_example, tokenize_examples


class DummyTokenizer:
    def __call__(self, text: str, **kwargs):
        tokens = [len(token) for token in text.split()]
        if kwargs.get("truncation"):
            max_length = int(kwargs.get("max_length", len(tokens)))
            tokens = tokens[:max_length]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }


def test_tokenize_report_matches_expected():
    root_dir = Path(__file__).resolve().parents[2]
    data_file = root_dir / "tests" / "data" / "testdata_de_en_1000.jsonl"
    report = TokenizeReport.from_path(root_dir / "tokenize_report.txt")
    ds = load_dataset("json", data_files=str(data_file), split="train")

    it = tokenize_examples(
        ds,
        tokenizer=DummyTokenizer(),
        tokenize_reporter=report,
        tokenizer_kwargs={"truncation": True, "max_length": 256},
    )

    try:
        for _ in it:
            pass
    finally:
        report.close()
        actual = root_dir / "tokenize_report.txt"
        expected = root_dir / "tests" / "expected" / "tokenizer" / "tokenize_report.txt"
        assert actual.read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")


def test_tokenize_example_adds_expected_structure():
    ex = {
        "id": "sample-1",
        "translation": {
            "de": "eins zwei drei vier",
            "en": "one two three four",
        },
    }

    out = tokenize_example(
        ex,
        tokenizer=DummyTokenizer(),
        tokenizer_kwargs={"truncation": True, "max_length": 3},
    )

    assert out["id"] == "sample-1"
    assert out["translation"] == ex["translation"]
    assert "tokenized_translation" in out

    de_tok = out["tokenized_translation"]["de"]
    en_tok = out["tokenized_translation"]["en"]

    assert de_tok["input_ids"] == [4, 4, 4]
    assert de_tok["attention_mask"] == [1, 1, 1]
    assert en_tok["input_ids"] == [3, 3, 5]
    assert en_tok["attention_mask"] == [1, 1, 1]

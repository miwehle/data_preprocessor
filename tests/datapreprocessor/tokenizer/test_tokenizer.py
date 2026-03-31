from pathlib import Path

from datasets import load_dataset

from data_preprocessor.tokenizer import TokenizeReport, tokenize_example, tokenize_examples


class DummyTokenizer:
    def __call__(self, text: str, **kwargs):
        tokens = [len(token) for token in text.split()]
        if kwargs.get("truncation"):
            max_length = int(kwargs.get("max_length", len(tokens)))
            tokens = tokens[:max_length]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


def test_tokenize_report_matches_expected():
    root_dir = Path(__file__).resolve().parents[3]
    data_file = root_dir / "tests" / "data" / "testdata_de_en_1000.jsonl"
    report = TokenizeReport.from_path(root_dir / "tokenize_report.txt")
    ds = load_dataset("json", data_files=str(data_file), split="train")

    it = tokenize_examples(
        ds,
        tokenizer=DummyTokenizer(),
        tokenize_reporter=report,
        tokenizer_kwargs=None,
    )

    try:
        for _ in it:
            pass
    finally:
        report.close()
        actual = root_dir / "tokenize_report.txt"
        expected = root_dir / "tests" / "expected" / "tokenizer" / "tokenize_report.txt"
        expected_text = expected.read_text(encoding="utf-8").replace("\\n", "\n")
        assert actual.read_text(encoding="utf-8") == expected_text


def test_tokenize_example_adds_expected_structure():
    ex = {
        "id": "sample-1",
        "translation": {"de": "eins zwei drei vier", "en": "one two three four"},
    }

    out = tokenize_example(
        ex, tokenizer=DummyTokenizer(), tokenizer_kwargs={"truncation": True, "max_length": 3}
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


def test_tokenize_examples_removes_too_long():
    examples = [
        {"id": 1, "translation": {"de": "eins zwei drei vier", "en": "one two"}},
        {"id": 2, "translation": {"de": "eins zwei", "en": "one two three"}},
        {"id": 3, "translation": {"de": "eins zwei drei", "en": "one two"}},
        {"id": 4, "translation": {"de": "eins zwei", "en": "one two"}},
    ]
    report_path = Path(".local_tmp") / "tokenize_report_remove_case.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = TokenizeReport.from_path(report_path)

    try:
        actual = list(
            tokenize_examples(
                examples,
                tokenizer=DummyTokenizer(),
                tokenize_reporter=report,
                max_seq_len=3,
            )
        )
    finally:
        report.close()

    assert [example["id"] for example in actual] == [3, 4]
    assert report_path.read_text(encoding="utf-8") == (
        "{'seq_no': 1, 'removed_id': 1}\n"
        "{'seq_no': 2, 'removed_id': 2}\n"
        "{'seq_no': 3, 'token_lengths': {'de': 3, 'en': 2}}\n"
        "{'seq_no': 4, 'token_lengths': {'de': 2, 'en': 2}}\n"
    )

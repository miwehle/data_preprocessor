from pathlib import Path
from functools import partial

import pytest
from datasets import load_dataset

from data_preprocessor.filter import FlawReport, filter_examples, keep
from data_preprocessor.filter.predicates import pair_predicates, predicates
from data_preprocessor.filter.predicates.text_pair_predicates import are_equal, bad_length_ratio
from data_preprocessor.filter.predicates.text_predicates import is_blank, is_too_short


def test_filter_with_keep():
    root_dir = Path(__file__).resolve().parents[3]
    data_file = root_dir / "tests" / "data" / "testdata_de_en_1000.jsonl"
    report = FlawReport.from_path(root_dir / "flaw_report.txt")
    ds = load_dataset("json", data_files=str(data_file), split="train")
    it = filter_examples(
        ds,
        partial(
            keep,
            flaw_reporter=report,
            text_flaws=predicates(
                [
                    "is_blank",
                    ["is_too_short", {"min": 10}],
                    ["is_too_long", {"max": 300}],
                    "contains_url",
                    "contains_email",
                    "contains_control_chars",
                    "contains_invisible_format_chars",
                    "has_odd_number_of_quotes",
                    "has_unbalanced_brackets",
                ]
            ),
            pair_flaws=pair_predicates(
                [["bad_length_ratio", {"min": 0.33, "max": 3}], "are_equal"]
            ),
        ),
    )

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


def test_predicates_resolves_yaml_entries():
    ps = predicates(["is_blank", ["is_too_short", {"min": 10}]])
    pps = pair_predicates([["bad_length_ratio", {"min": 0.33, "max": 3}], "are_equal"])

    assert ps[0] is is_blank
    assert isinstance(ps[1], partial)
    assert ps[1].func is is_too_short
    assert ps[1].keywords == {"min": 10}
    assert isinstance(pps[0], partial)
    assert pps[0].func is bad_length_ratio
    assert pps[0].keywords == {"min": 0.33, "max": 3}
    assert pps[1] is are_equal


def test_predicates_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown predicate"):
        predicates(["does_not_exist"])

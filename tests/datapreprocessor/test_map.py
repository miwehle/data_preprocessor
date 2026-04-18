from data_preprocessor import MapConfig
from data_preprocessor.map import map_examples


def test_map_examples_maps_required_fields():
    ds = [
        {
            "id": 5,
            "translation": {"de": "hallo", "en": "hello"},
            "tokenized_translation": {
                "de": {"input_ids": [1, 10, 2], "attention_mask": [1, 1, 1]},
                "en": {"input_ids": [1, 20, 2], "attention_mask": [1, 1, 1]},
            },
        }
    ]

    out = list(map_examples(ds, MapConfig(src_lang="de", tgt_lang="en")))

    assert out == [{"id": 5, "src_ids": [1, 10, 2], "tgt_ids": [1, 20, 2]}]


def test_map_examples_can_include_text():
    ds = [
        {
            "id": 7,
            "translation": {"de": "guten tag", "en": "good day"},
            "tokenized_translation": {
                "de": {"input_ids": [1, 30, 31, 2]},
                "en": {"input_ids": [1, 40, 41, 2]},
            },
        }
    ]

    out = list(map_examples(ds, MapConfig(src_lang="de", tgt_lang="en", include_text=True)))

    assert out == [
        {
            "id": 7,
            "src_ids": [1, 30, 31, 2],
            "tgt_ids": [1, 40, 41, 2],
            "src_text": "guten tag",
            "tgt_text": "good day",
        }
    ]


def test_map_examples_can_prepend_tgt_bos_and_ensure_tgt_eos():
    ds = [
        {
            "id": 9,
            "translation": {"de": "guten morgen", "en": "good morning"},
            "tokenized_translation": {"de": {"input_ids": [10, 11, 0]}, "en": {"input_ids": [40, 41, 0]}},
        }
    ]

    out = list(map_examples(ds, MapConfig(src_lang="de", tgt_lang="en", tgt_bos_id=99, tgt_eos_id=0)))

    assert out == [{"id": 9, "src_ids": [10, 11, 0], "tgt_ids": [99, 40, 41, 0]}]

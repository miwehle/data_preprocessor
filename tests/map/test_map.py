from datapreprocessor.map import to_training_schema


def test_to_training_schema_maps_required_fields():
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

    out = list(to_training_schema(ds))

    assert out == [{"id": 5, "src_ids": [1, 10, 2], "tgt_ids": [1, 20, 2]}]


def test_to_training_schema_can_include_text():
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

    out = list(to_training_schema(ds, include_text=True))

    assert out == [
        {
            "id": 7,
            "src_ids": [1, 30, 31, 2],
            "tgt_ids": [1, 40, 41, 2],
            "src_text": "guten tag",
            "tgt_text": "good day",
        }
    ]

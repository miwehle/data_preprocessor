from __future__ import annotations

from data_preprocessor import api


def test_map_projects_to_training_schema():
    out = list(
        api.map(
            [
                {
                    "id": 1,
                    "translation": {"de": "hallo", "en": "hello"},
                    "tokenized_translation": {
                        "de": {"input_ids": [1, 10, 2], "attention_mask": [1, 1, 1]},
                        "en": {"input_ids": [1, 20, 2], "attention_mask": [1, 1, 1]},
                    },
                }
            ],
            api.MapConfig(src_lang="de", tgt_lang="en", include_text=True),
        )
    )

    assert out == [{"id": 1, "src_ids": [1, 10, 2], "tgt_ids": [1, 20, 2], "src_text": "hallo", "tgt_text": "hello"}]


def test_map_can_write_target_bos_and_eos():
    out = list(
        api.map(
            [
                {
                    "id": 1,
                    "translation": {"de": "hallo", "en": "hello"},
                    "tokenized_translation": {"de": {"input_ids": [10, 0]}, "en": {"input_ids": [20, 0]}},
                }
            ],
            api.MapConfig(src_lang="de", tgt_lang="en", tgt_bos_id=99, tgt_eos_id=0),
        )
    )

    assert out == [{"id": 1, "src_ids": [10, 0], "tgt_ids": [99, 20, 0]}]

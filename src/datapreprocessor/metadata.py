from __future__ import annotations


def build_dataset_meta(
    *,
    tokenizer_model_name: str,
    src_lang: str,
    tgt_lang: str,
    id_field: str,
    src_field: str = "src_ids",
    tgt_field: str = "tgt_ids",
    base_vocab_size: int,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_id: int,
    tgt_pad_id: int,
    tgt_bos_id: int,
    tgt_eos_id: int,
    num_examples: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "tokenizer_model_name": tokenizer_model_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "id_field": id_field,
        "src_field": src_field,
        "tgt_field": tgt_field,
        "base_vocab_size": base_vocab_size,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "src_pad_id": src_pad_id,
        "tgt_pad_id": tgt_pad_id,
        "tgt_bos_id": tgt_bos_id,
        "tgt_eos_id": tgt_eos_id,
        "num_examples": num_examples,
    }

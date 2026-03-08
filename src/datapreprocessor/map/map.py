from __future__ import annotations

from collections.abc import Iterable, Iterator

from datapreprocessor.types import Example


def to_training_schema(
    ds: Iterable[Example],
    *,
    id_key: str = "id",
    tokenized_key: str = "tokenized_translation",
    src_lang: str = "de",
    tgt_lang: str = "en",
    include_text: bool = False,
) -> Iterator[Example]:
    """Project tokenized examples to a flat training schema for translation."""
    for ex in ds:
        tokenized = ex[tokenized_key]
        out: Example = {
            "id": int(ex[id_key]),
            "src_ids": list(tokenized[src_lang]["input_ids"]),
            "tgt_ids": list(tokenized[tgt_lang]["input_ids"]),
        }
        if include_text:
            translation = ex["translation"]
            out["src_text"] = str(translation[src_lang])
            out["tgt_text"] = str(translation[tgt_lang])
        yield out

from __future__ import annotations

from collections.abc import Iterable, Iterator

from data_preprocessor.shared import Example, MapConfig


def _normalize_target_ids(
    input_ids: list[int], *, tgt_bos_id: int | None, tgt_eos_id: int | None
) -> list[int]:
    """Ensure IDs start with BOS and end with EOS when configured."""
    normalized = [int(x) for x in input_ids]

    if tgt_bos_id is not None:
        if not normalized or normalized[0] != tgt_bos_id:
            normalized = [tgt_bos_id, *normalized]

    if tgt_eos_id is not None:
        if not normalized or normalized[-1] != tgt_eos_id:
            normalized.append(tgt_eos_id)

    return normalized


def map_examples(ds: Iterable[Example], config: MapConfig) -> Iterator[Example]:
    """Map tokenized pipeline examples to the flat translation training schema.

    This is the core function of the ``map`` stage: it turns nested
    ``tokenized_translation`` output into the ``id/src_ids/tgt_ids`` structure
    expected by Translator2 training.

    If needed, it also normalizes target token ID sequences for training by
    adding an explicit target BOS token at the front and ensuring a target EOS
    token at the end.

    Example:
    tokenized ``{"de": {"input_ids": [10, 11]}, "en": {"input_ids": [20, 0]}}``
    becomes ``{"src_ids": [10, 11], "tgt_ids": [99, 20, 0]}`` when
    ``tgt_bos_id=99`` and ``tgt_eos_id=0``.
    """
    for ex in ds:
        tokenized = ex[config.tokenized_key]
        src_ids = [int(x) for x in tokenized[config.src_lang]["input_ids"]]
        tgt_ids = _normalize_target_ids(
            list(tokenized[config.tgt_lang]["input_ids"]),
            tgt_bos_id=config.tgt_bos_id,
            tgt_eos_id=config.tgt_eos_id,
        )
        out: Example = {"id": int(ex[config.id_key or "id"]), "src_ids": src_ids, "tgt_ids": tgt_ids}
        if config.include_text:
            translation = ex["translation"]
            out["src_text"] = str(translation[config.src_lang])
            out["tgt_text"] = str(translation[config.tgt_lang])
        yield out

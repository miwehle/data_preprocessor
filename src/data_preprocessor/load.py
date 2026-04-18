from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasets import Dataset, load_dataset

from data_preprocessor.shared import Example, LoadConfig


def attach_ids(
    records: Dataset | Iterable[Example],
    *,
    include_ids: bool = True,
    id_field: str = "id",
    start_id: int = 0,
    overwrite_ids: bool = False,
) -> Dataset | list[Example]:
    if not include_ids:
        return records

    if isinstance(records, Dataset):
        if id_field in records.column_names:
            if not overwrite_ids:
                raise ValueError(
                    f"ID field '{id_field}' already exists. " "Set overwrite_ids=True to replace it."
                )
            records = records.remove_columns(id_field)

        ids = list(range(start_id, start_id + len(records)))
        return records.add_column(id_field, ids)

    out: list[Example] = []
    for idx, ex in enumerate(records, start=start_id):
        enriched: dict[str, Any] = dict(ex)
        enriched[id_field] = idx
        out.append(enriched)
    return out


def load_examples(config: LoadConfig) -> Dataset | list[Example]:
    """Load one dataset split from Hugging Face and optionally attach IDs."""
    ds = load_dataset(path=config.path_name, name=config.name, split=config.split, data_files=config.data_files)
    records = ds if config.max_examples is None else ds.select(range(min(config.max_examples, len(ds))))
    return attach_ids(
        records,
        include_ids=config.include_ids,
        id_field=config.id_field,
        start_id=config.start_id,
        overwrite_ids=config.overwrite_ids,
    )

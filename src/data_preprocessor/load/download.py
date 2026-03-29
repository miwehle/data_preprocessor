from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from datasets import Dataset, load_dataset

from data_preprocessor.types import Example


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
                    f"ID field '{id_field}' already exists. "
                    "Set overwrite_ids=True to replace it."
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


def download_examples(
    *,
    dataset: str,
    config: str,
    split: str,
    source_format: Literal["hf", "hf_parquet"] = "hf",
    revision: str = "main",
    parquet_basename: str | None = None,
    max_examples: int | None = None,
    include_ids: bool = True,
    id_field: str = "id",
    start_id: int = 0,
    overwrite_ids: bool = False,
) -> Dataset | list[Example]:
    """Load one dataset split from Hugging Face and optionally attach IDs.

    Supported source formats:
    - ``hf``: load via ``load_dataset(dataset, config, split=...)``.
    - ``hf_parquet``: load a split from a Hub-hosted parquet file.

    The ``hf_parquet`` mode is intended for datasets such as
    ``IWSLT/iwslt2017`` whose legacy dataset script is not supported by
    newer ``datasets`` versions.

    Args:
        dataset: Hugging Face dataset name, for example ``Helsinki-NLP/europarl``.
        config: Dataset config name, for example ``de-en`` or ``iwslt2017-de-en``.
        split: Split name such as ``train``, ``validation``, or ``test``.
        source_format: Dataset loading mode, either ``hf`` or ``hf_parquet``.
        revision: Hub revision used for ``hf_parquet`` downloads.
        parquet_basename: Optional parquet filename prefix override.
        max_examples: Optional cap on the number of loaded examples.
        include_ids: Whether to add sequential example IDs.
        id_field: Output field name for attached IDs.
        start_id: First generated ID value.
        overwrite_ids: Whether to replace an existing ID field.

    Returns:
        A dataset-like collection of examples, optionally enriched with IDs.

    Raises:
        ValueError: If ``source_format`` is unsupported.
        ValueError: If the ID field already exists and ``overwrite_ids`` is false.
    """    
    if source_format == "hf":
        ds = load_dataset(dataset, config, split=split)
    elif source_format == "hf_parquet":
        basename = parquet_basename or dataset.rsplit("/", maxsplit=1)[-1]
        ds = load_dataset(
            "parquet",
            data_files=(
                f"https://huggingface.co/datasets/{dataset}/resolve/"
                f"{revision}/{config}/{basename}-{split}.parquet"
            ),
            split="train",
        )
    else:
        raise ValueError(
            f"Unsupported source_format {source_format!r}. "
            "Expected 'hf' or 'hf_parquet'."
        )
    records = ds if max_examples is None else ds.select(range(min(max_examples, len(ds))))
    return attach_ids(
        records,
        include_ids=include_ids,
        id_field=id_field,
        start_id=start_id,
        overwrite_ids=overwrite_ids,
    )

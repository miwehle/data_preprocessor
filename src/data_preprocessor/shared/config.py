from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class DownloadConfig:
    path_name: str
    name: str | None = None
    split: str
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    dataset_name: str | None = None
    max_examples: int | None = None
    include_ids: bool = True
    id_field: str = "id"
    start_id: int = 0
    overwrite_ids: bool = False


@dataclass(frozen=True, kw_only=True)
class NormConfig:
    changes: list[Any] | None = None
    norm_debug: bool = False


@dataclass(frozen=True, kw_only=True)
class FilterConfig:
    predicates: list[Any] | None = None
    pair_predicates: list[Any] | None = None


@dataclass(frozen=True, kw_only=True)
class TokenizeConfig:
    tokenizer_model_name: str
    tokenizer_kwargs: dict | None = None
    tokenize_debug: bool = False
    max_seq_len: int | None = None
    src_lang: str | None = None


@dataclass(frozen=True, kw_only=True)
class MapConfig:
    src_lang: str
    tgt_lang: str
    id_key: str | None = None
    tokenized_key: str = "tokenized_translation"
    tgt_bos_id: int | None = None
    tgt_eos_id: int | None = None
    include_text: bool = False


@dataclass(frozen=True, kw_only=True)
class SplitConfig:
    dataset: str | None = None
    split_ratio: dict[str, float]
    seed: int

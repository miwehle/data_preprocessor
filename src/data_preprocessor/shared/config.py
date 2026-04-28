from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

_CONFIG = ConfigDict(extra="forbid")


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class LoadConfig:
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


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class NormConfig:
    changes: list[Any] | None = None
    norm_debug: bool = False


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class FilterConfig:
    predicates: list[Any] | None = None
    pair_predicates: list[Any] | None = None


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class TokenizeConfig:
    tokenizer_model_name: str
    tokenizer_kwargs: dict | None = None
    tokenize_debug: bool = False
    max_seq_len: int | None = None
    src_lang: str | None = None


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class MapConfig:
    src_lang: str
    tgt_lang: str
    id_key: str | None = None
    tokenized_key: str = "tokenized_translation"
    tgt_bos_id: int | None = None
    tgt_eos_id: int | None = None
    include_text: bool = False


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class SplitConfig:
    dataset: str | None = None
    split_ratio: dict[str, float]
    seed: int


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class PreprocessRunConfig:
    load_config: LoadConfig
    tokenize_config: TokenizeConfig
    map_config: MapConfig
    norm_config: NormConfig | None = None
    filter_config: FilterConfig | None = None
    split_config: SplitConfig | None = None
    artifacts_dir: str | Path | None = None
    staging_dir: str | Path | None = None
    write_snapshots: bool = False

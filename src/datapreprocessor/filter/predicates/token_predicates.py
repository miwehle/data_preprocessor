# token_checks.py
from __future__ import annotations

from typing import Sequence


def is_empty(tokens: Sequence[int]) -> bool:
    return len(tokens) == 0


def is_too_short(tokens: Sequence[int], *, min_len: int) -> bool:
    return len(tokens) < min_len


def is_too_long(tokens: Sequence[int], *, max_len: int) -> bool:
    return len(tokens) > max_len


def has_only_special_tokens(tokens: Sequence[int], *, special_ids: set[int]) -> bool:
    return all(t in special_ids for t in tokens)

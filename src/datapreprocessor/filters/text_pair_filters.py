# pair_filters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import text_filters as tf


TextPred = Callable[[str], bool]
PairPred = Callable[[str, str], bool]


@dataclass(frozen=True)
class LengthRatioConfig:
    """
    Ratio check on lengths.
    - unit="chars": uses len(text)
    - unit="tokens": you provide token_len_fn
    """
    min_ratio: float  # de/en lower bound
    max_ratio: float  # de/en upper bound
    unit: str = "chars"  # "chars" or "tokens"


def either(text_de: str, text_en: str, pred: TextPred) -> bool:
    return pred(text_de) or pred(text_en)


def both(text_de: str, text_en: str, pred: TextPred) -> bool:
    return pred(text_de) and pred(text_en)


def is_bad_any_text_filter(text_de: str, text_en: str, preds: Iterable[TextPred]) -> bool:
    return any(either(text_de, text_en, p) for p in preds)


def bad_length_ratio(
    text_de: str,
    text_en: str,
    cfg: LengthRatioConfig,
    *,
    token_len_fn: Optional[Callable[[str], int]] = None,
) -> bool:
    """
    Returns True if length(de)/length(en) is outside [min_ratio, max_ratio].

    Notes:
    - If either side has length 0 -> True (bad).
    - If unit="tokens" you must pass token_len_fn.
    """
    if cfg.min_ratio <= 0 or cfg.max_ratio <= 0 or cfg.max_ratio < cfg.min_ratio:
        raise ValueError("Invalid ratio bounds")

    if cfg.unit == "chars":
        de_len = len(text_de)
        en_len = len(text_en)
    elif cfg.unit == "tokens":
        if token_len_fn is None:
            raise ValueError("token_len_fn is required when unit='tokens'")
        de_len = token_len_fn(text_de)
        en_len = token_len_fn(text_en)
    else:
        raise ValueError("unit must be 'chars' or 'tokens'")

    if de_len == 0 or en_len == 0:
        return True

    ratio = de_len / en_len
    return not (cfg.min_ratio <= ratio <= cfg.max_ratio)


# Convenience pair predicates (built from text_filters)

def contains_url_either(text_de: str, text_en: str) -> bool:
    return either(text_de, text_en, tf.contains_url)


def contains_email_either(text_de: str, text_en: str) -> bool:
    return either(text_de, text_en, tf.contains_email)


def contains_controls_either(text_de: str, text_en: str) -> bool:
    return either(text_de, text_en, tf.contains_control_chars)


def contains_invisible_format_either(text_de: str, text_en: str) -> bool:
    return either(text_de, text_en, tf.contains_invisible_format_chars)


def is_blank_either(text_de: str, text_en: str) -> bool:
    return either(text_de, text_en, tf.is_blank)
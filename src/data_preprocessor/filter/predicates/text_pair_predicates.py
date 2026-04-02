"""Pair predicates addressable by name from YAML config via getattr().

Only keep intentionally YAML-exposed callables in this module.
"""

from __future__ import annotations
import builtins
from random import random


def bad_length_ratio(de: str, en: str, min, max) -> bool:
    if len(de) == 0 or len(en) == 0:
        return True

    r = len(de) / builtins.max(1, len(en))
    if r < min or r > max:
        return True

def are_equal(de, en):
    return de.lower() == en.lower()

def reject_with_probability(de, en, prob):
    return random() < prob

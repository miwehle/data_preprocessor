from __future__ import annotations
import builtins
from functools import partial


def bad_length_ratio(de: str, en: str, min, max) -> bool:
    if len(de) == 0 or len(en) == 0:
        return True

    r = len(de) / builtins.max(1, len(en))
    if r < min or r > max:
        return True


def are_equal(de, en):
    return de.lower() == en.lower()


TEXT_PAIR_FLAWS = (partial(bad_length_ratio, min=0.33, max=3), are_equal)

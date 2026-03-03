from __future__ import annotations

from functools import partial

from . import text_predicates as te
from . import text_pair_predicates as tep

TEXT_FLAWS = [
    te.is_blank,
    partial(te.is_too_short, min_chars=10),
    partial(te.is_too_long, max_chars=300),
    te.contains_url,
    te.contains_email,
    #te.contains_german_chars,
    te.contains_control_chars,
    te.contains_invisible_format_chars,
    te.has_odd_number_of_quotes,
    te.has_unbalanced_brackets,
]

TEXT_PAIR_FLAWS = [
    partial(tep.bad_length_ratio, min=0.33, max=3),
    tep.are_equal
]

TOKEN_FLAWS = [
]

TOKEN_PAIR_FLAWS = [
]

def find_flaws(flaws, *args):
    
    # Simplifies how matched predicates are represented in reports/logs.
    def format_flaw(f):
        if isinstance(f, partial):
            name = f.func.__name__
            parts = []
            if f.args:
                parts.extend(repr(a) for a in f.args)
            if f.keywords:
                parts.extend(f"{k}={v!r}" for k, v in f.keywords.items())
            return f"{name}({', '.join(parts)})"
        return f.__name__

    return [format_flaw(f) for f in flaws if f(*args)]


def check(x, flaws):
    return find_flaws(flaws, x)


def check_pair(x, y, flaws):
    return find_flaws(flaws, x, y)

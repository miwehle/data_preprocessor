from __future__ import annotations

from functools import partial

from .filters import text_filters as te
from .filters import text_pair_filters as tep

#from .filters import token_filters as to
#from .filters import token_filters as top

TEXT_FILTERS = [
    te.is_blank,
    partial(te.is_too_short, min_chars=10),
    partial(te.is_too_long, max_chars=300),
    te.contains_url,
    te.contains_email,
    te.contains_german_chars,
    te.contains_control_chars,
    te.contains_invisible_format_chars,
    te.has_odd_number_of_quotes,
    te.has_unbalanced_brackets,
]

TEXT_PAIR_FILTERS = [
    partial(tep.bad_length_ratio, min=0.33, max=3),
    tep.are_equal
]

TOKEN_FILTERS = [
]

TOKEN_PAIR_FILTERS = [
]

def violations(filters, *args):
    return [f for f in filters if f(*args)]


def check(x, filters):
    return violations(filters, x)


def check_pair(x, y, filters):
    return violations(filters, x, y)
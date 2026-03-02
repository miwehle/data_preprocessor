from __future__ import annotations

from .filters import text_filters as te
from .filters import text_pair_filters as tep

from .filters import token_filters as to
from .filters import token_filters as top

TEXT_FILTERS = [
    te.contains_url,
    te.contains_email
]

TEXT_PAIR_FILTERS = [
    tep.bad_length_ratio
]

TOKEN_FILTERS = [
]

TOKEN_PAIR_FILTERS = [
]

def is_bad(x, filters):
    return any(f(x) for f in filters)

def is_bad_pair(x, y, filters):
    return any(f(x, y) for f in filters)
import re
from typing import Callable

_WHITESPACE_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_APOSTROPHE_SPACING_RE = re.compile(r"(?<=\w)['’]\s+(?=\w)")
_UNICODE_QUOTE_MAP = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‹": "'",
        "›": "'",
        "«": '"',
        "»": '"',
    }
)


def strip_edges(text: str) -> str:
    return text.strip()


def remove_control_chars(text: str) -> str:
    return _CTRL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text)


def normalize_unicode_quotes(text: str) -> str:
    return text.translate(_UNICODE_QUOTE_MAP)


def fix_apostrophe_spacing(text: str) -> str:
    return _APOSTROPHE_SPACING_RE.sub("'", text)


Change = Callable[[str], str]

CHANGES: tuple[Change, ...] = (
    strip_edges,
    remove_control_chars,
    collapse_whitespace,
    normalize_unicode_quotes,
    # fix_apostrophe_spacing,
)

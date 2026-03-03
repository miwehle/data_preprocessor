from __future__ import annotations

import re
import unicodedata
from typing import Iterable


# --- URL / Email -------------------------------------------------------------

_URL_RE = re.compile(
    r"""
    (?:
        (?:https?|ftp)://          # schema
        |
        www\.                      # www.
        |
        \b                         # word boundary
        (?:[A-Za-z0-9-]+\.)+       # domain parts
        (?:[A-Za-z]{2,})           # tld
    )
    """,
    re.VERBOSE,
)

_EMAIL_RE = re.compile(
    r"""
    \b
    [A-Za-z0-9._%+-]+
    @
    [A-Za-z0-9.-]+
    \.
    [A-Za-z]{2,}
    \b
    """,
    re.VERBOSE,
)

_GERMAN_CHARS = re.compile(r"[äöüÄÖÜß]")

def contains_url(text: str) -> bool:
    return _URL_RE.search(text) is not None


def contains_email(text: str) -> bool:
    return _EMAIL_RE.search(text) is not None

def contains_german_chars(text: str) -> bool:
    return _GERMAN_CHARS.search(text)


# --- Whitespace / length -----------------------------------------------------

def is_blank(text: str) -> bool:
    return text.strip() == ""


def is_too_short(text: str, *, min_chars: int) -> bool:
    if min_chars < 0:
        raise ValueError("min_chars must be >= 0")
    return len(text) < min_chars


def is_too_long(text: str, *, max_chars: int) -> bool:
    if max_chars < 0:
        raise ValueError("max_chars must be >= 0")
    return len(text) > max_chars


# --- Unicode oddities --------------------------------------------------------

def contains_control_chars(text: str, *, allow: str = "\t\n\r") -> bool:
    """
    True if text contains Unicode control chars (category Cc),
    excluding those explicitly allowed (default: tab/newline/carriage-return).
    """
    allow_set = set(allow)
    for ch in text:
        if ch in allow_set:
            continue
        if unicodedata.category(ch) == "Cc":
            return True
    return False


def contains_invisible_format_chars(text: str) -> bool:
    """
    True if text contains Unicode "format" characters (category Cf),
    e.g. zero-width joiner/non-joiner, bidi marks.
    """
    return any(unicodedata.category(ch) == "Cf" for ch in text)


# --- Quotes / brackets (simple heuristics) -----------------------------------

_QUOTE_CHARS = {
    '"', "'", "„", "“", "”", "‚", "‘", "’", "«", "»",
}
_BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "«": "»",
    "‹": "›",
    "“": "”",
    "„": "“",  # common DE usage
    "‘": "’",
    "‚": "‘",
}


def has_odd_number_of_quotes(text: str, *, quote_chars: Iterable[str] = _QUOTE_CHARS) -> bool:
    """
    Heuristic. Counts quote characters and flags odd totals.
    Not language-smart (doesn't handle apostrophes in contractions, etc.).
    """
    qs = set(quote_chars)
    count = sum(1 for ch in text if ch in qs)
    return (count % 2) == 1


def has_unbalanced_brackets(text: str) -> bool:
    """
    Simple stack-based check for common bracket-like pairs.
    """
    stack: list[str] = []
    openers = set(_BRACKET_PAIRS.keys())
    closers = set(_BRACKET_PAIRS.values())

    inverse = {v: k for k, v in _BRACKET_PAIRS.items()}

    for ch in text:
        if ch in openers:
            stack.append(ch)
        elif ch in closers:
            if not stack:
                return True
            opener = stack.pop()
            if inverse.get(ch) != opener:
                return True

    return len(stack) != 0
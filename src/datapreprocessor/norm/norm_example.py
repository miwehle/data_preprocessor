import re
from pathlib import Path
from typing import Any, Callable, Dict, Protocol, TextIO

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
Change = Callable[[str], str]
Example = Dict[str, Any]


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


CHANGES: list[Change] = [
    strip_edges,
    remove_control_chars,
    collapse_whitespace,
    normalize_unicode_quotes,
    fix_apostrophe_spacing,
]


def apply_changes(text: str, changes: list[Change] = CHANGES) -> tuple[str, list[str]]:
    change_names: list[str] = []
    current = text
    for change in changes:
        updated = change(current)
        if updated != current:
            change_names.append(change.__name__)
        current = updated
    return current, change_names


class NormReporter(Protocol):
    def note_change(self, before: str, after: str, norm_changes: list[str]) -> None: ...


class NormReport:
    def __init__(self, out: TextIO, *, debug: bool = False):
        self.out = out
        self.debug = debug
        self.seq_no = 0

    @classmethod
    def from_path(cls, path: str | Path = "norm_report.txt", *, debug: bool = False) -> "NormReport":
        return cls(open(path, "w", encoding="utf-8"), debug=debug)

    def note_change(self, before: str, after: str, norm_changes: list[str]) -> None:
        self.seq_no += 1
        if not norm_changes:
            return
        record = {
            "seq_no": self.seq_no,
            "norm_changes": norm_changes,
        }
        if self.debug:
            record["before"] = before
            record["after"] = after
        self.out.write(f"{record}\n")

    def flush(self) -> None:
        self.out.flush()

    def close(self) -> None:
        self.out.close()


def norm(s: str, norm_reporter: NormReporter | None = None) -> str:
    """Normalize text by removing control chars and collapsing whitespace."""
    before = str(s)
    after, norm_changes = apply_changes(before)

    if norm_reporter is not None:
        norm_reporter.note_change(before, after, norm_changes)

    return after


def norm_example(ex: Example, norm_reporter: NormReporter | None = None) -> Example:
    """Return a normalized copy of one translation example with de/en texts."""
    normalized = dict(ex)
    translation = dict(normalized["translation"])
    translation["de"] = norm(translation["de"], norm_reporter=norm_reporter)
    translation["en"] = norm(translation["en"], norm_reporter=norm_reporter)
    normalized["translation"] = translation
    return normalized

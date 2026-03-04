from functools import partial
from pathlib import Path
from typing import Protocol, TextIO

from .filter import Example
from .predicates import text_pair_predicates as tep
from .predicates import text_predicates as te

TEXT_FLAWS = [
    te.is_blank,
    partial(te.is_too_short, min_chars=10),
    partial(te.is_too_long, max_chars=300),
    te.contains_url,
    te.contains_email,
    # te.contains_german_chars,
    te.contains_control_chars,
    te.contains_invisible_format_chars,
    te.has_odd_number_of_quotes,
    te.has_unbalanced_brackets,
]

TEXT_PAIR_FLAWS = [
    partial(tep.bad_length_ratio, min=0.33, max=3),
    tep.are_equal,
]

TOKEN_FLAWS = []
TOKEN_PAIR_FLAWS = []


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


class FlawReporter(Protocol):
    def note_flaws(self, de_flaws, en_flaws, pair_flaws) -> None: ...


class FlawReport:
    def __init__(self, out: TextIO):
        self.out = out
        self.seq_no = 0

    @classmethod
    def from_path(cls, path: str | Path = "flaw_report.txt") -> "FlawReport":
        return cls(open(path, "w", encoding="utf-8"))

    def note_flaws(self, de_flaws, en_flaws, pair_flaws):
        self.seq_no += 1

        if (de_flaws != [] or en_flaws != [] or pair_flaws != []):
            record = {
                "seq_no": self.seq_no,
                "de_flaws": de_flaws,
                "en_flaws": en_flaws,
                "pair_flaws": pair_flaws,
            }
            self.out.write(f"{record}\n")

    def flush(self) -> None:
        self.out.flush()

    def close(self) -> None:
        self.out.close()


def keep(ex: Example, flaw_reporter: FlawReporter | None = None):
    """Return True for clean examples and optionally report flaw findings."""
    def check(x, flaws):
        return find_flaws(flaws, x)

    def check_pair(x, y, flaws):
        return find_flaws(flaws, x, y)

    de = ex["translation"]["de"]
    en = ex["translation"]["en"]

    de_flaws = check(de, TEXT_FLAWS)
    en_flaws = check(en, TEXT_FLAWS)
    pair_flaws = check_pair(de, en, TEXT_PAIR_FLAWS)

    if flaw_reporter is not None:
        flaw_reporter.note_flaws(de_flaws, en_flaws, pair_flaws)

    return de_flaws == [] and en_flaws == [] and pair_flaws == []

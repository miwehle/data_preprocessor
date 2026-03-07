from functools import partial
from pathlib import Path
from typing import Protocol, TextIO

from .filter import Example
from .predicates.text_pair_predicates import TEXT_PAIR_FLAWS
from .predicates.text_predicates import TEXT_FLAWS


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
    def from_path(cls, path: str | Path) -> "FlawReport":
        return cls(open(path, "w", encoding="utf-8"))

    def note_flaws(self, de_flaws, en_flaws, pair_flaws):
        self.seq_no += 1

        if de_flaws != [] or en_flaws != [] or pair_flaws != []:
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


def keep(
    ex: Example,
    text_flaws=TEXT_FLAWS,
    pair_flaws=TEXT_PAIR_FLAWS,
    flaw_reporter: FlawReporter | None = None,
):
    """Return True for clean examples and optionally report flaw findings."""

    def check(x, flaws):
        return find_flaws(flaws, x)

    def check_pair(x, y, flaws):
        return find_flaws(flaws, x, y)

    de = ex["translation"]["de"]
    en = ex["translation"]["en"]

    de_flaws = check(de, text_flaws)
    en_flaws = check(en, text_flaws)
    pair_flaw_hits = check_pair(de, en, pair_flaws)

    if flaw_reporter is not None:
        flaw_reporter.note_flaws(de_flaws, en_flaws, pair_flaw_hits)

    return de_flaws == [] and en_flaws == [] and pair_flaw_hits == []

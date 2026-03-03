from pathlib import Path
from typing import TextIO, Protocol

from .filter import Example
from .check import check, check_pair, TEXT_FLAWS, TEXT_PAIR_FLAWS


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

    de = ex["translation"]["de"]
    en = ex["translation"]["en"]

    de_flaws = check(de, TEXT_FLAWS)
    en_flaws = check(en, TEXT_FLAWS)
    pair_flaws = check_pair(de, en, TEXT_PAIR_FLAWS)

    if flaw_reporter is not None:
        flaw_reporter.note_flaws(de_flaws, en_flaws, pair_flaws)

    return de_flaws == [] and en_flaws == [] and pair_flaws == []

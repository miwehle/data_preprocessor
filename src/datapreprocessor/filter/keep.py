from typing import TextIO

from .filter import Example
from .check import check, check_pair, TEXT_FLAWS, TEXT_PAIR_FLAWS

class FlawReport:
    _singleton = None

    def __init__(self, out: TextIO):
        self.out = out
        self.seq_no = 0

    @classmethod
    def singleton(cls):
        if cls._singleton is None:
            cls._singleton = cls(open("flaw_report.txt", "w", encoding="utf-8"))
        return cls._singleton

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


def keep(ex: Example):
    """Return True for clean examples.

    Side effect: write flaw findings to flaw_report.txt in the current working directory.
    """

    de = ex["translation"]["de"]
    en = ex["translation"]["en"]

    de_flaws = check(de, TEXT_FLAWS)
    en_flaws = check(en, TEXT_FLAWS)
    pair_flaws = check_pair(de, en, TEXT_PAIR_FLAWS)

    FlawReport.singleton().note_flaws(de_flaws, en_flaws, pair_flaws)

    return de_flaws == [] and en_flaws == [] and pair_flaws == []

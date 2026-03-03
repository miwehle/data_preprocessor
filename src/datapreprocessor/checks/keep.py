# ANWENDUNG

from typing import TextIO

from datasets import load_dataset
from .filter import filtered_examples, Example
from .check import check, check_pair, TEXT_FLAWS, TEXT_PAIR_FLAWS

print("Start!")

class FlawReport:
    def __init__(self, out: TextIO):
        self.out = out
        self.seq_no = 0

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

report_out: TextIO = open("../flaw_report.log", "w", encoding="utf-8")
report = FlawReport(report_out)

def keep(ex: Example):
    # dein Filterkriterium (hier Dummy)

    de = ex["translation"]["de"]
    en = ex["translation"]["en"]

    de_flaws = check(de, TEXT_FLAWS)
    en_flaws = check(en, TEXT_FLAWS)
    pair_flaws = check_pair(de, en, TEXT_PAIR_FLAWS)

    report.note_flaws(de_flaws, en_flaws, pair_flaws)

    return de_flaws == [] and en_flaws == [] and pair_flaws == []



ds = load_dataset("json", data_files="../data/testdata_de_en_100.jsonl", split="train")
#ds = load_dataset("Helsinki-NLP/europarl", "de-en", split="train", streaming=True)
it = filtered_examples(ds, keep)

try:
    for ex in it:
        pass
finally:
    report_out.close()

print("Fertig.")

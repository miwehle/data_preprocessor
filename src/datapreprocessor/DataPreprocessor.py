#PV2
#import time
#t = time.time()

import os
import filter.keep as c

from norm import norm
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


class DataPreprocessor:
    """
    load
    
    # auf der internen Rep:
    filter   # inkl. norm

    # in Datei:
    save     # Formate: txt, arrow, json
    tokenize # Formate: dieselben? Besser separat (andere Klasse)
    """

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        self.data = None
        self.flaws = {}
    
    @classmethod
    def _check(cls, de: str, en: str) -> bool:
        flaws = c.find_flaws(c.TEXT_FLAWS, de)
        flaws += c.find_flaws(c.TEXT_FLAWS, en)
        flaws += c.find_flaws(c.TEXT_PAIR_FLAWS, de, en)
        return flaws

    def load_from_file(self, file):
        """Lädt Daten aus einer Datei"""
        """ soll die mit save gespeicherten Daten laden"""
        """ annahme sie wurden im Arrow-Format gespeichert """

    def load(self, num_pairs):
        """Lädt n Paare von Daten"""
        print("> load")
        dataset = load_dataset("Helsinki-NLP/europarl", "de-en", split="train")
        self.data = []
        for i in range(min(num_pairs, len(dataset))):
            item = dataset[i]
            self.data.append({
                "de": item["translation"]["de"],
                "en": item["translation"]["en"]
            })
        print("< load")

    def preprocess(self, block_size):
        """Wendet den Tokenizer auf die geladenen Paare an"""
        """Truncated auf block_size"""
        print("> preprocess")
        if self.data:
            tokenized_data = []
            for idx, item in enumerate(self.data):
                de = norm(item["de"])
                en = norm(item["en"])
                flaws = self._check(de, en)
                if len(flaws) == 0:
                  de_tokens = self.tok(item["de"], truncation=True, max_length=block_size)
                  en_tokens = self.tok(item["en"], truncation=True, max_length=block_size)
                  tokenized_data.append({
                      "de": de_tokens["input_ids"],
                      "en": en_tokens["input_ids"]
                  })
                else:
                    self.flaws[idx] = flaws
            self.data = tokenized_data
        print("< preprocess")

    def save(self, file):
        """Speichert die Daten mit save_to_disk (Arrow) oder to_json (JSONL)"""
        if self.data:
            dataset = Dataset.from_list(self.data)
            if file.endswith('.jsonl'):
                dataset.to_json(file)
            else:
                dataset.save_to_disk(file)

    def save_txt(self, dir):
        """Schreibt die Daten menschenlesbar in einen Ordner,
          gesplittet in Dateien à pairs_per_file."""
        if not self.data:
            return

        pairs_per_file=10_000
        os.makedirs(dir, exist_ok=True)

        file_idx = 0
        pair_count = 0
        f = None

        try:
            for i, item in enumerate(self.data, 1):

                # Neue Datei öffnen, wenn nötig
                if pair_count == 0:
                    path = os.path.join(dir, f"{file_idx:04d}.txt")
                    f = open(path, "w", encoding="utf-8")

                f.write(f"- {i-1} -\n")
                f.write(f"DE: {item['de']}\n")
                f.write(f"EN: {item['en']}\n\n")

                pair_count += 1

                # Datei schließen und nächste vorbereiten
                if pair_count == pairs_per_file:
                    f.close()
                    f = None
                    file_idx += 1
                    pair_count = 0

        finally:
            if f is not None:
                f.close()

    def main(self):
        self.load(num_pairs=10000)
        self.save_txt("data/txt/europarl")
        self.preprocess(256)
        self.save("data/arrow/europarl.arrow")

if __name__ == "__main__":
    #print("Import:", time.time() - t)
    preprocessor = DataPreprocessor()
    preprocessor.main()

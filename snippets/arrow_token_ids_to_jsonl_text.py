import json

from datasets import load_from_disk
from transformers import AutoTokenizer

dataset_path = datasets_dir / "europarl_examples_for_comet" / "curated_dataset_split-validation"
output_path = datasets_dir / "europarl_examples_for_comet" / "dataset.jsonl"


tokenizer_name = "Helsinki-NLP/opus-mt-de-en"
dataset = load_from_disk(str(dataset_path))
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

with output_path.open("w", encoding="utf-8") as f:
    for row in dataset:
        record = {
            "src": tokenizer.decode(row["src_ids"], skip_special_tokens=True).strip(),
            "ref": tokenizer.decode(row["tgt_ids"], skip_special_tokens=True).strip(),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(output_path)

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))
from ipy import ops


def _artifacts_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / "tests" / ".test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def test_ops_map_projects_to_training_schema_jsonl():
    root = _artifacts_dir()
    src = root / f"{uuid4().hex}.jsonl"
    dst = root / f"{uuid4().hex}.jsonl"
    _write_jsonl(
        src,
        [
            {
                "id": 1,
                "translation": {"de": "hallo", "en": "hello"},
                "tokenized_translation": {
                    "de": {"input_ids": [1, 10, 2], "attention_mask": [1, 1, 1]},
                    "en": {"input_ids": [1, 20, 2], "attention_mask": [1, 1, 1]},
                },
            }
        ],
    )

    ops.map(input_path=src, output_path=dst, include_text=True)

    out = _read_jsonl(dst)
    assert out == [
        {
            "id": 1,
            "src_ids": [1, 10, 2],
            "tgt_ids": [1, 20, 2],
            "src_text": "hallo",
            "tgt_text": "hello",
        }
    ]

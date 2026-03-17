"""Run the dataset preprocessing pipeline from the command line.

This script receives a YAML config file. The config file is passed through to
`preprocess(...)` via its stage-specific `*_cfg` dictionaries.

Example: `configs/europarl_config.yaml`
Run it with: `python scripts/preprocess.py configs/europarl_config.yaml`

Datasets are stored under `artifacts/datasets`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


OPTIONAL_CFG_KEYS = ("download_cfg", "norm_cfg", "filter_cfg", "tokenize_cfg", "map_cfg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datapreprocessor.api import preprocess


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/preprocess.py <config-path>")
        return 1

    config_path = Path(sys.argv[1])
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return 1

    try:
        preprocess(
            write_jsonl=cfg.get("write_jsonl", True),
            **{key: cfg.get(key) for key in OPTIONAL_CFG_KEYS},
        )
    except Exception as exc:
        print(f"Preprocess failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Call the `preprocess` function from `src/api.py`.

Parameters:
- This script (`scripts/preprocess.py`) receives a YAML config file.
- The config file defines the dataset, config, and split.
- Nothing else needs to be specified; `preprocess` uses its default values.
- Example: `configs/europarl_config.yaml`
- Run it with: `python scripts/preprocess.py configs/europarl_config.yaml`

Storage location for the downloaded raw data:
- Next to the project directory, there is an `artifacts` directory.
- Datasets are stored under `artifacts/datasets`.
- Store raw data in a subdirectory that roughly matches the dataset name.
- Example: `artifacts/datasets/europarl/raw`

Storage location for the fully preprocessed dataset:
- Example: `artifacts/datasets/europarl/preprocessed`
- Store the metadata required for the model and training in the same directory.

Storage location for intermediate results:
- Example: `artifacts/datasets/europarl/interim`
- Do not create separate directories for steps such as `norm` or `filtered`;
  encode the step name in the filename instead.
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
            dataset=cfg["dataset"],  # required
            config=cfg["config"],
            split=cfg["split"],
            write_jsonl=cfg.get("write_jsonl", True),
            **{key: cfg.get(key) for key in OPTIONAL_CFG_KEYS},
        )
    except Exception as exc:
        print(f"Preprocess failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

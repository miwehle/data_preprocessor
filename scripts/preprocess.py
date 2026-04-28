"""Run the dataset preprocessing pipeline from the command line.

This script receives a YAML config file. The config file is passed through to
`preprocess(...)` via its stage-specific `*_config` dictionaries.

Example: `configs/europarl_config.yaml`
Run it with: `python scripts/preprocess.py configs/europarl_config.yaml`

Datasets are stored under `artifacts/datasets` unless `artifacts_dir` is set.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SHARED_SRC_DIR = REPO_ROOT.parent / "lab_infrastructure" / "src"
for path in (SRC_DIR, SHARED_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    from lab_infrastructure.run_config import read_run_config_as

    from data_preprocessor import PreprocessRunConfig, preprocess

    if len(sys.argv) != 2:
        print("Usage: python scripts/preprocess.py <config-path>")
        return 1

    try:
        cfg = read_run_config_as(Path(sys.argv[1]), PreprocessRunConfig)
        preprocess(cfg)
    except Exception as exc:
        print(f"Preprocess failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

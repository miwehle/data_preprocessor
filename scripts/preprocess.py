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
    from lab_infrastructure import run_config_cli

    from data_preprocessor import PreprocessRunConfig, preprocess

    run_config_cli(preprocess, PreprocessRunConfig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

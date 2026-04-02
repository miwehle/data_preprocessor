"""Run the dataset preprocessing pipeline from the command line.

This script receives a YAML config file. The config file is passed through to
`preprocess(...)` via its stage-specific `*_cfg` dictionaries.

Example: `configs/europarl_config.yaml`
Run it with: `python scripts/preprocess.py configs/europarl_config.yaml`

Datasets are stored under `artifacts/datasets` unless `artifacts_dir` is set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_preprocessor import (
    DownloadConfig,
    FilterConfig,
    MapConfig,
    NormConfig,
    TokenizeConfig,
    preprocess,
)


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
            download_cfg=DownloadConfig(**cfg["download_cfg"]),
            tokenize_cfg=TokenizeConfig(**cfg["tokenize_cfg"]),
            map_cfg=MapConfig(**cfg["map_cfg"]),
            norm_cfg=NormConfig(**cfg["norm_cfg"]) if cfg.get("norm_cfg") is not None else None,
            filter_cfg=FilterConfig(**cfg["filter_cfg"]) if cfg.get("filter_cfg") is not None else None,
            artifacts_dir=cfg.get("artifacts_dir"),
            staging_dir=cfg.get("staging_dir"),
            write_jsonl=cfg.get("write_jsonl", True),
        )
    except Exception as exc:
        print(f"Preprocess failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
SHARED_SRC = PROJECT_ROOT.parent / "lab_infrastructure" / "src"
TMP = PROJECT_ROOT / ".local_tmp"
HF_HOME = PROJECT_ROOT / ".hf_home"
HF_DATASETS_CACHE = HF_HOME / "datasets"

for path in (TMP, HF_HOME, HF_DATASETS_CACHE):
    path.mkdir(parents=True, exist_ok=True)

for path in (SRC, SHARED_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["TEMP"] = str(TMP)
os.environ["TMP"] = str(TMP)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)
tempfile.tempdir = str(TMP)


def pytest_configure(config) -> None:
    base_temp = TMP / "pytest"
    base_temp.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(base_temp)


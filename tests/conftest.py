from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from lab_infrastructure.logging import close_logger

# Keep Hugging Face datasets cache/temp paths inside the repo during tests.
# This avoids stalls/locks on global cache and temp dirs (e.g. on corporate
# machines where antivirus/endpoint protection may aggressively scan them).

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SHARED_SRC = _ROOT.parent / "lab_infrastructure" / "src"
_HF_HOME = _ROOT / ".hf_home"
_HF_DATASETS_CACHE = _HF_HOME / "datasets"
_TMP = _ROOT / ".local_tmp"

for p in (_HF_HOME, _HF_DATASETS_CACHE, _TMP):
    p.mkdir(parents=True, exist_ok=True)

for path in (_SRC, _SHARED_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_DATASETS_CACHE))
os.environ.setdefault("TEMP", str(_TMP))
os.environ.setdefault("TMP", str(_TMP))


@pytest.fixture(autouse=True)
def _close_data_preprocessor_logging_after_test():
    yield
    close_logger("data_preprocessor")


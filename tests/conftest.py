from __future__ import annotations

import os
from pathlib import Path

# Keep Hugging Face datasets cache/temp paths inside the repo during tests.
# This avoids stalls/locks on global cache and temp dirs (e.g. on corporate
# machines where antivirus/endpoint protection may aggressively scan them).

_ROOT = Path(__file__).resolve().parents[1]
_HF_HOME = _ROOT / ".hf_home"
_HF_DATASETS_CACHE = _HF_HOME / "datasets"
_TMP = _ROOT / ".tmp"

for p in (_HF_HOME, _HF_DATASETS_CACHE, _TMP):
    p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_DATASETS_CACHE))
os.environ.setdefault("TEMP", str(_TMP))
os.environ.setdefault("TMP", str(_TMP))

# DEFINITION stream_or_save

from __future__ import annotations
from typing import Callable, Dict, Any, Iterable, Iterator, Optional
import datasets


Example = Dict[str, Any]
Predicate = Callable[[Example], bool]


def filtered_examples(ds: Iterable[Example], keep: Predicate) -> Iterator[Example]:
    """Core: liefert nur die Beispiele, die keep(example) passieren."""
    for ex in ds:
        if keep(ex):
            yield ex


def save_to_disk(
    it: Iterable[Example],
    out_dir: str,
    *,
    features: Optional[datasets.Features] = None,
) -> datasets.Dataset:
    """
    Materialisiert einen Iterator zu einem HF Dataset und speichert es.
    Gut, wenn du später wieder schnell mit load_from_disk arbeiten willst.
    """
    ds_out = datasets.Dataset.from_generator(lambda: it, features=features)
    ds_out.save_to_disk(out_dir)
    return ds_out
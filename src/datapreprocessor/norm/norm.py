from __future__ import annotations

from typing import Iterable, Iterator

from .norm_example import Example, NormReporter, norm_example


def norm_examples(ds: Iterable[Example], norm_reporter: NormReporter | None = None) -> Iterator[Example]:
    """Yield normalized examples from an input iterable."""
    for ex in ds:
        yield norm_example(ex, norm_reporter=norm_reporter)

from __future__ import annotations

from typing import Iterable, Iterator

from .tokenize_example import Example, TokenizeReporter, TokenizerLike, tokenize_example


def tokenize_examples(
    ds: Iterable[Example],
    tokenizer: TokenizerLike,
    tokenize_reporter: TokenizeReporter | None = None,
    **tokenize_example_kwargs,
) -> Iterator[Example]:
    """Yield tokenized examples from an input iterable."""
    for ex in ds:
        yield tokenize_example(
            ex, tokenizer=tokenizer, tokenize_reporter=tokenize_reporter, **tokenize_example_kwargs
        )

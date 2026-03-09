from __future__ import annotations

from typing import Iterable, Iterator

from .tokenize_example import Example, TokenizeReporter, TokenizerLike, tokenize_example


def create_hf_tokenizer(model_name: str) -> TokenizerLike:
    if "opus-mt-" in model_name.lower():
        try:
            import sacremoses  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Tokenizer model appears to be Marian/OPUS-MT, but 'sacremoses' "
                "is not installed. Install it with: "
                "python -m pip install sacremoses"
            ) from exc

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


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

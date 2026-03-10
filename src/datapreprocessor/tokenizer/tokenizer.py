from __future__ import annotations

from typing import Any, Iterable, Iterator

from .tokenize_example import Example, TokenizeReporter, Tokenizer, tokenize_example


def create_hf_tokenizer(model_name: str) -> Tokenizer:
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


def resolve_training_token_ids(tokenizer: Any) -> dict[str, int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for training output.")
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for training output.")

    if bos_token_id is None:
        max_special_id = max(
            int(x) for x in [eos_token_id, pad_token_id, unk_token_id] if x is not None
        )
        bos_token_id = max_special_id + 1

    return {
        "src_pad_id": int(pad_token_id),
        "tgt_pad_id": int(pad_token_id),
        "tgt_bos_id": int(bos_token_id),
        "tgt_eos_id": int(eos_token_id),
    }


def tokenize_examples(
    ds: Iterable[Example],
    tokenizer: Tokenizer,
    tokenize_reporter: TokenizeReporter | None = None,
    **tokenize_example_kwargs,
) -> Iterator[Example]:
    """Yield tokenized examples from an input iterable."""
    for ex in ds:
        yield tokenize_example(
            ex, tokenizer=tokenizer, tokenize_reporter=tokenize_reporter, **tokenize_example_kwargs
        )

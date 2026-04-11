from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from lab_infrastructure.logging import get_logger, log_calls

from ..shared import TokenizeConfig
from .tokenize_example import Example, TokenizeReporter, Tokenizer, tokenize_example

_log_calls = log_calls(
    get_logger("data_preprocessor", log_path=Path(__file__).resolve().parents[4] / "artifacts" / "data_preprocessor.log")
)


@_log_calls
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


@_log_calls
def resolve_training_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve training token IDs from the tokenizer and synthesize target BOS if missing."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for training output.")
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for training output.")

    if bos_token_id is None:
        max_special_id = max(int(x) for x in [eos_token_id, pad_token_id, unk_token_id] if x is not None)
        bos_token_id = max_special_id + 1

    return {
        "src_pad_id": int(pad_token_id),
        "tgt_pad_id": int(pad_token_id),
        "tgt_bos_id": int(bos_token_id),
        "tgt_eos_id": int(eos_token_id),
    }


def tokenize_examples(
    ds: Iterable[Example],
    config: TokenizeConfig,
    tokenizer: Tokenizer,
    tokenize_reporter: TokenizeReporter | None = None,
) -> Iterator[Example]:
    """Yield tokenized examples and optionally drop examples with overlong token sequences."""
    for ex in ds:
        tokenized = tokenize_example(ex, tokenizer=tokenizer, tokenizer_kwargs=config.tokenizer_kwargs)
        tokenized_translation = tokenized["tokenized_translation"]
        src_lang = config.src_lang or "de"
        src_seq_len = len(tokenized_translation[src_lang]["input_ids"])
        tgt_lang = next(lang for lang in tokenized_translation if lang != src_lang)
        tgt_seq_len = len(tokenized_translation[tgt_lang]["input_ids"])
        if config.max_seq_len is not None and (
            src_seq_len > config.max_seq_len or tgt_seq_len >= config.max_seq_len
        ):
            if tokenize_reporter is not None:
                tokenize_reporter.note_example_too_long(ex.get("id"))
            continue
        if tokenize_reporter is not None:
            tokenize_reporter.note_tokenization(
                {lang: len(data["input_ids"]) for lang, data in tokenized_translation.items()}
            )
        yield tokenized


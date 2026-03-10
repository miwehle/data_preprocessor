from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, TextIO

from ..types import Example


class Tokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


class TokenizeReporter(Protocol):
    def note_tokenization(self, token_lengths: dict[str, int]) -> None: ...


class TokenizeReport:
    def __init__(self, out: TextIO, *, debug: bool = False):
        self.out = out
        self.debug = debug
        self.seq_no = 0

    @classmethod
    def from_path(cls, path: str | Path, *, debug: bool = False) -> "TokenizeReport":
        return cls(open(path, "w", encoding="utf-8"), debug=debug)

    def note_tokenization(self, token_lengths: dict[str, int]) -> None:
        self.seq_no += 1
        record: dict[str, Any] = {"seq_no": self.seq_no, "token_lengths": token_lengths}
        self.out.write(f"{record}\n")

    def close(self) -> None:
        self.out.close()


def _to_plain_dict(tokenized: Mapping[str, Any]) -> Dict[str, Any]:
    plain: Dict[str, Any] = {}
    for key, value in tokenized.items():
        if hasattr(value, "tolist"):
            plain[key] = value.tolist()
        else:
            plain[key] = value
    return plain


def _token_len(tokenized: Mapping[str, Any]) -> int:
    input_ids = tokenized.get("input_ids")
    if isinstance(input_ids, list):
        return len(input_ids)
    return 0


def tokenize_example(
    ex: Example,
    tokenizer: Tokenizer,
    tokenize_reporter: TokenizeReporter | None = None,
    *,
    translation_key: str = "translation",
    text_keys: tuple[str, str] = ("de", "en"),
    output_key: str = "tokenized_translation",
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> Example:
    """Return a copy of one example with per-language tokenization results."""
    kwargs = dict(tokenizer_kwargs or {})

    result = dict(ex)
    translation = dict(result[translation_key])

    tokenized_translation: Dict[str, Dict[str, Any]] = {}
    token_lengths: dict[str, int] = {}

    for lang in text_keys:
        text = str(translation[lang])
        tokenized = _to_plain_dict(tokenizer(text, **kwargs))
        tokenized_translation[lang] = tokenized
        token_lengths[lang] = _token_len(tokenized)

    result[output_key] = tokenized_translation

    if tokenize_reporter is not None:
        tokenize_reporter.note_tokenization(token_lengths)

    return result

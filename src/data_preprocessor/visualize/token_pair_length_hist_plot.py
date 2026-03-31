from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

try:
    from . import _pair_length_hist_plot_base as base
except ImportError:
    import _pair_length_hist_plot_base as base

def _extract_token_pair_lengths(record: dict) -> tuple[int, int] | None:
    tokenized = record.get("tokenized_translation")
    if not isinstance(tokenized, dict):
        return None
    de, en = tokenized.get("de"), tokenized.get("en")
    if not isinstance(de, dict) or not isinstance(en, dict):
        return None
    de_ids, en_ids = de.get("input_ids"), en.get("input_ids")
    if not isinstance(de_ids, list) or not isinstance(en_ids, list):
        return None
    return len(de_ids), len(en_ids)

def load_token_pair_lengths(dataset_path: str | Path) -> tuple[list[int], list[int]]:
    return base.load_pair_lengths(dataset_path, _extract_token_pair_lengths)

def plot_token_pair_length_histogram(
    dataset_path: str | Path,
    max_bins: int = 60,
    x_scale: str = "linear",
    y_scale: str = "linear",
    interactive_scale_toggle: bool = True,
    show_loading: bool = False,
):
    title = "Histogram of Token Pair Lengths (de/en)"
    fig = ax = progress = None
    if show_loading:
        fig, ax, progress = base.pu.show_loading_plot(title)
    de_lengths, en_lengths = base.load_pair_lengths(
        dataset_path, _extract_token_pair_lengths, progress
    )
    return base.plot_pair_length_histogram(
        de_lengths, en_lengths, title, "Text length (tokens)", max_bins,
        x_scale, y_scale, interactive_scale_toggle, fig, ax,
    )

def main() -> None:
    run("../artifacts/datasets/iwslt2017_iwslt2017-de-en_train_staging (3)/iwslt2017_tokenized.jsonl")

def run(dataset_path: str | Path) -> None:
    plot_token_pair_length_histogram(dataset_path, show_loading=True)
    plt.show()

if __name__ == "__main__":
    main()

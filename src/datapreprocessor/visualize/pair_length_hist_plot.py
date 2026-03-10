from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from . import plot_utils as pu
except ImportError:
    import plot_utils as pu


def _extract_pair(record: dict) -> tuple[str, str] | None:
    translation = record.get("translation")
    if isinstance(translation, dict):
        de = translation.get("de")
        en = translation.get("en")
    else:
        de = record.get("de")
        en = record.get("en")

    if de is None or en is None:
        return None
    return str(de), str(en)


def load_pair_lengths(dataset_path: str | Path) -> tuple[list[int], list[int]]:
    path = Path(dataset_path)
    de_lengths: list[int] = []
    en_lengths: list[int] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no} in {path}") from exc

                pair = _extract_pair(record)
                if pair is None:
                    continue
                de, en = pair
                de_lengths.append(len(de))
                en_lengths.append(len(en))
    else:
        from datasets import load_from_disk

        ds = load_from_disk(str(path))
        for record in ds:
            pair = _extract_pair(record)
            if pair is None:
                continue
            de, en = pair
            de_lengths.append(len(de))
            en_lengths.append(len(en))

    return de_lengths, en_lengths


def plot_pair_length_histogram(
    dataset_path: str | Path,
    *,
    max_bins: int = 60,
    x_scale: str = "linear",
    y_scale: str = "linear",
    interactive_scale_toggle: bool = True,
):
    de_lengths, en_lengths = load_pair_lengths(dataset_path)

    fig, ax = plt.subplots(figsize=(11, 6))
    if not de_lengths or not en_lengths:
        ax.text(0.5, 0.5, "No de/en pairs found", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    bins = pu.integer_histogram_bins(de_lengths, en_lengths, max_bins=max_bins)
    ax.hist(
        de_lengths,
        bins=bins,
        alpha=0.55,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.25,
        label="de",
    )
    ax.hist(
        en_lengths,
        bins=bins,
        alpha=0.45,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.25,
        label="en",
    )
    ax.set_title("Histogram of Pair Lengths (de/en)")
    ax.set_xlabel("Text length (characters)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    pu.set_x_axis_scale(ax, x_scale)
    pu.set_y_axis_scale(ax, y_scale)
    if interactive_scale_toggle:
        pu.attach_x_scale_toggle(fig, ax, key="x")
        pu.attach_y_scale_toggle(fig, ax, key="y")
        pu.attach_toolbar_hint(fig, "Press 'x' or 'y' to toggle scaling")
    pu.set_coord_display(ax)
    fig.tight_layout()

    return fig, ax


def main() -> None:
    run("data/europarl/raw/europarl_de-en_train_1000.jsonl")


def run(dataset_path: str | Path) -> None:
    plot_pair_length_histogram(dataset_path)
    plt.show()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from . import plot_utils as pu
except ImportError:
    import plot_utils as pu

def _records(dataset_path: str | Path, progress=None):
    path = Path(dataset_path)
    if path.suffix.lower() != ".jsonl":
        if progress is not None:
            progress(None)
        from datasets import load_from_disk
        yield from load_from_disk(str(path))
        if progress is not None:
            progress(1.0)
        return
    total = max(path.stat().st_size, 1)
    read_bytes = 0
    if progress is not None:
        progress(0.0)
    with path.open("rb") as f:
        for line_no, line in enumerate(f, 1):
            read_bytes += len(line)
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}") from exc
            if progress is not None and line_no % 2000 == 0:
                progress(read_bytes / total)
    if progress is not None:
        progress(1.0)

def load_pair_lengths(
    dataset_path: str | Path, extract_lengths, progress=None
) -> tuple[list[int], list[int]]:
    de_lengths, en_lengths = [], []
    for record in _records(dataset_path, progress):
        lengths = extract_lengths(record)
        if lengths is None:
            continue
        de_lengths.append(lengths[0])
        en_lengths.append(lengths[1])
    return de_lengths, en_lengths

def plot_pair_length_histogram(
    de_lengths: list[int],
    en_lengths: list[int],
    title: str,
    x_label: str,
    max_bins: int = 60,
    x_scale: str = "linear",
    y_scale: str = "linear",
    interactive_scale_toggle: bool = True,
    fig=None,
    ax=None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        ax.clear()
    if not de_lengths or not en_lengths:
        ax.text(0.5, 0.5, "No de/en pairs found", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    bins = pu.integer_histogram_bins(de_lengths, en_lengths, max_bins=max_bins)
    for lengths, alpha, color, label in (
        (de_lengths, 0.55, "#1f77b4", "de"),
        (en_lengths, 0.45, "#ff7f0e", "en"),
    ):
        ax.hist(
            lengths, bins, alpha=alpha, color=color, edgecolor="black",
            linewidth=0.25, label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
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

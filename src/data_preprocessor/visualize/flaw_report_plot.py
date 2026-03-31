from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from . import plot_utils as pu
except ImportError:
    import plot_utils as pu


def _count_flaws(records: list[dict]) -> tuple[Counter, Counter, Counter]:
    de_counts: Counter = Counter()
    en_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for record in records:
        de_counts.update(record.get("de_flaws", []))
        en_counts.update(record.get("en_flaws", []))
        pair_counts.update(record.get("pair_flaws", []))

    return de_counts, en_counts, pair_counts


def plot_flaw_counts(report_path: str | Path, show_loading: bool = False):
    """
    Build two separate figures:
    1) grouped bar chart for de_flaws and en_flaws
    2) separate bar chart for pair_flaws
    """
    fig_text = ax_text = progress = None
    if show_loading:
        fig_text, ax_text, progress = pu.show_loading_plot("Flaw Counts: de_flaws and en_flaws")
    records = pu.load_report_records(report_path, progress)
    de_counts, en_counts, pair_counts = _count_flaws(records)

    text_flaws = sorted(
        set(de_counts.keys()) | set(en_counts.keys()),
        key=lambda flaw: de_counts.get(flaw, 0) + en_counts.get(flaw, 0),
        reverse=True,
    )
    pair_flaws = sorted(pair_counts.keys(), key=lambda flaw: pair_counts.get(flaw, 0), reverse=True)

    if fig_text is None or ax_text is None:
        fig_text, ax_text = plt.subplots(figsize=(12, 6))
    else:
        ax_text.clear()
    fig_pair, ax_pair = plt.subplots(figsize=(12, 6))

    pu.plot_grouped_category_counts(
        fig_text, ax_text, text_flaws, de_counts, en_counts,
        "de_flaws", "en_flaws", "No de/en flaws found",
    )

    ax_text.set_title("Flaw Counts: de_flaws and en_flaws")
    ax_text.set_xlabel("Flaw")
    ax_text.set_ylabel("Count")
    pu.set_coord_display(ax_text)
    fig_text.tight_layout()
    pu.attach_adaptive_xtick_labels(fig_text, ax_text)

    if pair_flaws:
        x_pair = list(range(len(pair_flaws)))
        pair_bars = ax_pair.bar(
            x_pair, [pair_counts.get(f, 0) for f in pair_flaws], width=0.6, label="pair_flaws"
        )
        ax_pair.set_xticks(x_pair)
        ax_pair.set_xticklabels(
            [pu.format_wrapped_label(f) for f in pair_flaws], rotation=0, ha="center"
        )
        ax_pair.legend()
        pu.add_headroom(ax_pair, pair_bars)
        pair_annotations = pu.annotate_bars(ax_pair, pair_bars)
        pu.attach_adaptive_value_labels(fig_pair, list(pair_bars), pair_annotations)
    else:
        ax_pair.text(0.5, 0.5, "No pair flaws found", ha="center", va="center")
        ax_pair.set_axis_off()

    ax_pair.set_title("Flaw Counts: pair_flaws")
    ax_pair.set_xlabel("Flaw")
    ax_pair.set_ylabel("Count")
    pu.set_coord_display(ax_pair)
    fig_pair.tight_layout()
    pu.attach_adaptive_xtick_labels(fig_pair, ax_pair)

    return (fig_text, ax_text), (fig_pair, ax_pair)


def run(report_path: str | Path = "flaw_report.txt") -> None:
    plot_flaw_counts(report_path, show_loading=True)
    plt.show()


def main() -> None:
    run("../artifacts/datasets/iwslt2017_iwslt2017-de-en_train_staging (3)/flaw_report.txt")


if __name__ == "__main__":
    main()

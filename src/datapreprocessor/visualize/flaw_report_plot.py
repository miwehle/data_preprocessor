from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from . import plot_utils as pu
except ImportError:
    import plot_utils as pu


def _load_records(report_path: str | Path) -> list[dict]:
    path = Path(report_path)
    records: list[dict] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = ast.literal_eval(line)
        if isinstance(record, dict):
            records.append(record)

    return records


def _count_flaws(records: list[dict]) -> tuple[Counter, Counter, Counter]:
    de_counts: Counter = Counter()
    en_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for record in records:
        de_counts.update(record.get("de_flaws", []))
        en_counts.update(record.get("en_flaws", []))
        pair_counts.update(record.get("pair_flaws", []))

    return de_counts, en_counts, pair_counts


def plot_flaw_counts(report_path: str | Path):
    """
    Build two separate figures:
    1) grouped bar chart for de_flaws and en_flaws
    2) separate bar chart for pair_flaws
    """
    records = _load_records(report_path)
    de_counts, en_counts, pair_counts = _count_flaws(records)

    text_flaws = sorted(
        set(de_counts.keys()) | set(en_counts.keys()),
        key=lambda flaw: de_counts.get(flaw, 0) + en_counts.get(flaw, 0),
        reverse=True,
    )
    pair_flaws = sorted(pair_counts.keys(), key=lambda flaw: pair_counts.get(flaw, 0), reverse=True)

    fig_text, ax_text = plt.subplots(figsize=(12, 6))
    fig_pair, ax_pair = plt.subplots(figsize=(12, 6))

    if text_flaws:
        x_text = list(range(len(text_flaws)))
        width = 0.35

        de_bars = ax_text.bar(
            [i - width / 2 for i in x_text],
            [de_counts.get(f, 0) for f in text_flaws],
            width=width,
            label="de_flaws",
        )
        en_bars = ax_text.bar(
            [i + width / 2 for i in x_text],
            [en_counts.get(f, 0) for f in text_flaws],
            width=width,
            label="en_flaws",
        )
        ax_text.set_xticks(x_text)
        ax_text.set_xticklabels(
            [pu.format_wrapped_label(f) for f in text_flaws], rotation=0, ha="center"
        )
        ax_text.legend()
        pu.add_headroom(ax_text, list(de_bars) + list(en_bars))
        de_annotations = pu.annotate_bars(ax_text, de_bars)
        en_annotations = pu.annotate_bars(ax_text, en_bars)
        pu.attach_adaptive_value_labels(
            fig_text, list(de_bars) + list(en_bars), de_annotations + en_annotations
        )
    else:
        ax_text.text(0.5, 0.5, "No de/en flaws found", ha="center", va="center")
        ax_text.set_axis_off()

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
    plot_flaw_counts(report_path)
    plt.show()


def main() -> None:
    run("data/europarl/reports/flaw_report.txt")


if __name__ == "__main__":
    main()

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt


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


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _add_headroom(ax, bars, ratio: float = 0.10) -> None:
    max_height = max((bar.get_height() for bar in bars), default=0)
    if max_height <= 0:
        return
    ax.set_ylim(0, max_height * (1 + ratio))


def _format_flaw_label(label: str, width: int = 22) -> str:
    text = label.replace("_", " ")
    text = re.sub(r"(?<=\S)\(", " (", text)
    text = re.sub(r"\s+", " ", text).strip()
    return fill(text, width=width)


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
        ax_text.set_xticklabels([_format_flaw_label(f) for f in text_flaws], rotation=0, ha="center")
        ax_text.legend()
        _add_headroom(ax_text, list(de_bars) + list(en_bars))
        _annotate_bars(ax_text, de_bars)
        _annotate_bars(ax_text, en_bars)
    else:
        ax_text.text(0.5, 0.5, "No de/en flaws found", ha="center", va="center")
        ax_text.set_axis_off()

    ax_text.set_title("Flaw Counts: de_flaws and en_flaws")
    ax_text.set_xlabel("Flaw")
    ax_text.set_ylabel("Count")
    fig_text.tight_layout()

    if pair_flaws:
        x_pair = list(range(len(pair_flaws)))
        pair_bars = ax_pair.bar(x_pair, [pair_counts.get(f, 0) for f in pair_flaws], width=0.6, label="pair_flaws")
        ax_pair.set_xticks(x_pair)
        ax_pair.set_xticklabels([_format_flaw_label(f) for f in pair_flaws], rotation=0, ha="center")
        ax_pair.legend()
        _add_headroom(ax_pair, pair_bars)
        _annotate_bars(ax_pair, pair_bars)
    else:
        ax_pair.text(0.5, 0.5, "No pair flaws found", ha="center", va="center")
        ax_pair.set_axis_off()

    ax_pair.set_title("Flaw Counts: pair_flaws")
    ax_pair.set_xlabel("Flaw")
    ax_pair.set_ylabel("Count")
    fig_pair.tight_layout()

    return (fig_text, ax_text), (fig_pair, ax_pair)

from __future__ import annotations

import re
from textwrap import fill


def annotate_bars(ax, bars):
    annotations = []
    for bar in bars:
        height = bar.get_height()
        ann = ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        annotations.append(ann)
    return annotations


def add_headroom(ax, bars, ratio: float = 0.10) -> None:
    max_height = max((bar.get_height() for bar in bars), default=0)
    if max_height <= 0:
        return
    ax.set_ylim(0, max_height * (1 + ratio))


def format_wrapped_label(label: str, width: int = 22) -> str:
    text = label.replace("_", " ")
    text = re.sub(r"(?<=\S)\(", " (", text)
    text = re.sub(r"\s+", " ", text).strip()
    return fill(text, width=width)


def _needs_diagonal_xtick_labels(ax, renderer) -> bool:
    labels = [label for label in ax.get_xticklabels() if label.get_text()]
    if len(labels) <= 1:
        return False

    axis_width_px = ax.get_window_extent(renderer=renderer).width
    if axis_width_px <= 0:
        return False

    slot_width_px = axis_width_px / len(labels)
    max_label_width_px = 0.0
    for label in labels:
        width_px = label.get_window_extent(renderer=renderer).width
        max_label_width_px = max(max_label_width_px, width_px)

    return max_label_width_px > slot_width_px * 0.95


def _set_xtick_label_orientation(ax, diagonal: bool) -> None:
    rotation = 45 if diagonal else 0
    ha = "right" if diagonal else "center"
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha(ha)


def attach_adaptive_xtick_labels(fig, ax) -> None:
    state = {"diagonal": None}

    def _update(_event=None):
        renderer = fig.canvas.get_renderer()
        diagonal = _needs_diagonal_xtick_labels(ax, renderer)
        if diagonal != state["diagonal"]:
            _set_xtick_label_orientation(ax, diagonal)
            state["diagonal"] = diagonal
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _update)
    fig.canvas.draw()
    _update()


def _needs_vertical_value_labels(bars, annotations, renderer) -> bool:
    if not bars or not annotations:
        return False
    for bar, ann in zip(bars, annotations):
        bar_width_px = bar.get_window_extent(renderer=renderer).width
        label_width_px = ann.get_window_extent(renderer=renderer).width
        if label_width_px > bar_width_px * 0.9:
            return True
    return False


def _set_value_label_orientation(annotations, vertical: bool) -> None:
    rotation = 90 if vertical else 0
    for ann in annotations:
        ann.set_rotation(rotation)
        ann.set_ha("center")
        ann.set_va("bottom")


def attach_adaptive_value_labels(fig, bars, annotations) -> None:
    state = {"vertical": None}

    def _update(_event=None):
        renderer = fig.canvas.get_renderer()
        vertical = _needs_vertical_value_labels(bars, annotations, renderer)
        if vertical != state["vertical"]:
            _set_value_label_orientation(annotations, vertical)
            state["vertical"] = vertical
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _update)
    fig.canvas.draw()
    _update()


def set_coord_display(ax) -> None:
    def _fmt(v: float) -> str:
        s = f"{v:.2f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    ax.format_coord = lambda x, y: f"(x, y) = ({_fmt(x)}, {_fmt(y)})"

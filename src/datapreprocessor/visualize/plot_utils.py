from __future__ import annotations

import math
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
    if len(text) <= width:
        return text

    def _wrap_keep_newlines(s: str) -> str:
        return "\n".join(fill(part, width=width) for part in s.splitlines())

    # Default wrapping fallback.
    wrapped_default = fill(text, width=width)

    # Prefer wrap opportunities before opening brackets when wrapping is needed.
    bracket_break_candidates: list[str] = []
    for match in re.finditer(r" \(", text):
        i = match.start()
        candidate = _wrap_keep_newlines(text[:i] + "\n" + text[i + 1 :])
        bracket_break_candidates.append(candidate)

    if not bracket_break_candidates:
        return wrapped_default

    def _score(s: str) -> tuple[int, int]:
        lines = s.splitlines()
        too_long = sum(1 for line in lines if len(line) > width)
        max_len = max((len(line) for line in lines), default=0)
        return too_long, max_len

    feasible = [c for c in bracket_break_candidates if _score(c)[0] == 0]
    if feasible:
        return min(feasible, key=_score)

    return wrapped_default


def _max_xtick_label_width_ratio(ax, renderer) -> float:
    labels = [label for label in ax.get_xticklabels() if label.get_text()]
    if len(labels) <= 1:
        return 0.0

    axis_width_px = ax.get_window_extent(renderer=renderer).width
    if axis_width_px <= 0:
        return 0.0

    slot_width_px = axis_width_px / len(labels)
    max_label_width_px = 0.0
    for label in labels:
        width_px = label.get_window_extent(renderer=renderer).width
        max_label_width_px = max(max_label_width_px, width_px)

    return max_label_width_px / slot_width_px


def _set_xtick_label_orientation(ax, diagonal: bool) -> None:
    rotation = 45 if diagonal else 0
    ha = "right" if diagonal else "center"
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha(ha)


def attach_adaptive_xtick_labels(fig, ax) -> None:
    # Hysteresis to avoid oscillation in transition widths.
    enter_diagonal_ratio = 0.95
    exit_diagonal_ratio = 0.80
    state = {"diagonal": None}

    def _update(_event=None):
        renderer = fig.canvas.get_renderer()
        ratio = _max_xtick_label_width_ratio(ax, renderer)
        if state["diagonal"] is None:
            diagonal = ratio > enter_diagonal_ratio
        elif state["diagonal"]:
            diagonal = ratio > exit_diagonal_ratio
        else:
            diagonal = ratio > enter_diagonal_ratio
        if diagonal != state["diagonal"]:
            _set_xtick_label_orientation(ax, diagonal)
            state["diagonal"] = diagonal
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _update)
    fig.canvas.draw()
    _update()


def _max_value_label_width_ratio(bars, annotations, renderer) -> float:
    if not bars or not annotations:
        return 0.0

    max_ratio = 0.0
    for bar, ann in zip(bars, annotations):
        bar_width_px = bar.get_window_extent(renderer=renderer).width
        if bar_width_px <= 0:
            continue
        # Measure in horizontal orientation, independent from current rotation.
        label_width_px, _, _ = renderer.get_text_width_height_descent(
            ann.get_text(), ann.get_fontproperties(), ismath=False
        )
        max_ratio = max(max_ratio, label_width_px / bar_width_px)

    return max_ratio


def _set_value_label_orientation(annotations, vertical: bool) -> None:
    rotation = 90 if vertical else 0
    for ann in annotations:
        ann.set_rotation(rotation)
        ann.set_ha("center")
        ann.set_va("bottom")


def attach_adaptive_value_labels(fig, bars, annotations) -> None:
    # Hysteresis to avoid oscillation in transition widths.
    enter_vertical_ratio = 0.90
    exit_vertical_ratio = 0.75
    state = {"vertical": None}

    def _update(_event=None):
        renderer = fig.canvas.get_renderer()
        ratio = _max_value_label_width_ratio(bars, annotations, renderer)
        if state["vertical"] is None:
            vertical = ratio > enter_vertical_ratio
        elif state["vertical"]:
            vertical = ratio > exit_vertical_ratio
        else:
            vertical = ratio > enter_vertical_ratio

        if vertical != state["vertical"]:
            _set_value_label_orientation(annotations, vertical)
            state["vertical"] = vertical
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _update)
    fig.canvas.draw()
    _update()


def set_coord_display(ax) -> None:
    manager = getattr(ax.figure.canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None)
    if toolbar is not None:
        message_label = getattr(toolbar, "_message_label", None)
        if message_label is not None:
            try:
                import tkinter.font as tkfont

                message_label.configure(font=tkfont.nametofont("TkFixedFont"))
            except Exception:
                pass

    def _width(bounds: tuple[float, float]) -> int:
        lo, hi = bounds
        return max(len(f"{lo:.2f}"), len(f"{hi:.2f}"))

    def _fmt(v: float, width: int) -> str:
        return f"{v:>{width}.2f}"

    ax.format_coord = lambda x, y: (
        f"{_fmt(x, _width(ax.get_xlim()))} {_fmt(y, _width(ax.get_ylim()))}"
    )


def attach_toolbar_hint(fig, text: str):
    manager = getattr(fig.canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None)
    if toolbar is None or hasattr(toolbar, "_codex_hint_label"):
        return None

    try:
        import tkinter as tk
    except ImportError:
        return None

    label = tk.Label(master=toolbar, text=text, font=getattr(toolbar, "_label_font", None), padx=8)
    label.pack(side=tk.LEFT)
    toolbar._codex_hint_label = label
    return label


def integer_histogram_bins(*value_series, max_bins: int = 60) -> list[float]:
    values = [int(v) for series in value_series for v in series]
    if not values:
        return [-0.5, 0.5]

    min_value = min(values)
    max_value = max(values)
    value_span = (max_value - min_value) + 1
    step = max(1, math.ceil(value_span / max_bins))

    start = min_value - 0.5
    n_bins = math.ceil(value_span / step)
    return [start + i * step for i in range(n_bins + 1)]


def _validate_scale(scale: str) -> None:
    if scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'")


def set_x_axis_scale(ax, scale: str) -> None:
    _validate_scale(scale)
    ax.set_xscale(scale)


def set_y_axis_scale(ax, scale: str) -> None:
    _validate_scale(scale)
    ax.set_yscale(scale)


def attach_axis_scale_toggle(fig, ax, *, axis: str, key: str):
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    scales = ("linear", "log")

    def toggle() -> str:
        current = ax.get_xscale() if axis == "x" else ax.get_yscale()
        if current not in scales:
            next_scale = "linear"
        else:
            next_scale = scales[(scales.index(current) + 1) % len(scales)]
        if axis == "x":
            set_x_axis_scale(ax, next_scale)
        else:
            set_y_axis_scale(ax, next_scale)
        fig.canvas.draw_idle()
        return next_scale

    def _on_key(event) -> None:
        pressed = (event.key or "").lower()
        if pressed == key.lower():
            toggle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    return toggle


def attach_x_scale_toggle(fig, ax, *, key: str = "x"):
    return attach_axis_scale_toggle(fig, ax, axis="x", key=key)


def attach_y_scale_toggle(fig, ax, *, key: str = "y"):
    return attach_axis_scale_toggle(fig, ax, axis="y", key=key)

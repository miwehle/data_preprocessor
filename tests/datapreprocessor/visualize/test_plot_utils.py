from datapreprocessor.visualize.plot_utils import format_wrapped_label
from datapreprocessor.visualize.plot_utils import attach_y_scale_toggle
from datapreprocessor.visualize.plot_utils import attach_x_scale_toggle
from datapreprocessor.visualize.plot_utils import set_y_axis_scale
from datapreprocessor.visualize.plot_utils import set_x_axis_scale
from datapreprocessor.visualize.plot_utils import integer_histogram_bins


def test_format_wrapped_label_prefers_break_before_opening_paren() -> None:
    label = "is_too_long(max_chars=500)"
    wrapped = format_wrapped_label(label, width=16)
    assert wrapped == "is too long\n(max chars=500)"


def test_format_wrapped_label_keeps_short_label_on_one_line() -> None:
    label = "contains_email"
    wrapped = format_wrapped_label(label, width=30)
    assert wrapped == "contains email"


def test_integer_histogram_bins_scales_step_to_cap_bin_count() -> None:
    bins = integer_histogram_bins([1, 2, 3], [100], max_bins=10)
    assert len(bins) <= 11
    assert bins[0] <= 0.5
    assert bins[-1] >= 100.5


def test_integer_histogram_bins_empty_fallback() -> None:
    bins = integer_histogram_bins([], [], max_bins=10)
    assert bins == [-0.5, 0.5]


def test_set_y_axis_scale_accepts_linear_and_log() -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    set_y_axis_scale(ax, "log")
    assert ax.get_yscale() == "log"
    set_y_axis_scale(ax, "linear")
    assert ax.get_yscale() == "linear"


def test_set_x_axis_scale_accepts_linear_and_log() -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    set_x_axis_scale(ax, "log")
    assert ax.get_xscale() == "log"
    set_x_axis_scale(ax, "linear")
    assert ax.get_xscale() == "linear"


def test_set_y_axis_scale_rejects_invalid_value() -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    try:
        set_y_axis_scale(ax, "invalid")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_set_x_axis_scale_rejects_invalid_value() -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    try:
        set_x_axis_scale(ax, "invalid")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_attach_y_scale_toggle_switches_between_linear_and_log() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    toggle = attach_y_scale_toggle(fig, ax, key="y")
    assert ax.get_yscale() == "linear"
    assert toggle() == "log"
    assert ax.get_yscale() == "log"
    assert toggle() == "linear"
    assert ax.get_yscale() == "linear"


def test_attach_x_scale_toggle_switches_between_linear_and_log() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    toggle = attach_x_scale_toggle(fig, ax, key="x")
    assert ax.get_xscale() == "linear"
    assert toggle() == "log"
    assert ax.get_xscale() == "log"
    assert toggle() == "linear"
    assert ax.get_xscale() == "linear"

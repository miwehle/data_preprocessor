import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from uuid import uuid4

from datapreprocessor.visualize.pair_length_hist_plot import load_pair_lengths
from datapreprocessor.visualize.pair_length_hist_plot import plot_pair_length_histogram


def _local_temp_jsonl(lines: list[str]) -> Path:
    path = Path("tests/data") / f"tmp_pair_lengths_{uuid4().hex}.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_load_pair_lengths_reads_translation_pairs() -> None:
    dataset_path = _local_temp_jsonl(
        [
            '{"translation": {"de": "Hallo", "en": "Hello"}}',
            '{"translation": {"de": "Welt", "en": "World!"}}',
            '{"foo": "bar"}',
        ]
    )
    try:
        de_lengths, en_lengths = load_pair_lengths(dataset_path)
    finally:
        dataset_path.unlink(missing_ok=True)

    assert de_lengths == [5, 4]
    assert en_lengths == [5, 6]


def test_plot_pair_length_histogram_returns_axis_with_two_histograms() -> None:
    dataset_path = _local_temp_jsonl(
        [
            '{"translation": {"de": "a", "en": "bb"}}',
            '{"translation": {"de": "ccc", "en": "dddd"}}',
        ]
    )

    try:
        _, ax = plot_pair_length_histogram(dataset_path, max_bins=5)
    finally:
        dataset_path.unlink(missing_ok=True)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]

    assert labels == ["de", "en"]
    assert len(ax.patches) > 0


def test_plot_pair_length_histogram_accepts_log_scale_on_call() -> None:
    dataset_path = _local_temp_jsonl(
        [
            '{"translation": {"de": "a", "en": "bb"}}',
            '{"translation": {"de": "ccc", "en": "dddd"}}',
        ]
    )
    try:
        _, ax = plot_pair_length_histogram(
            dataset_path,
            max_bins=5,
            y_scale="log",
            interactive_scale_toggle=False,
        )
    finally:
        dataset_path.unlink(missing_ok=True)

    assert ax.get_yscale() == "log"


def test_plot_pair_length_histogram_supports_all_axis_scale_combinations() -> None:
    dataset_path = _local_temp_jsonl(
        [
            '{"translation": {"de": "a", "en": "bb"}}',
            '{"translation": {"de": "ccc", "en": "dddd"}}',
            '{"translation": {"de": "eeeee", "en": "ffffff"}}',
        ]
    )
    combinations = [
        ("linear", "linear"),
        ("log", "linear"),
        ("linear", "log"),
        ("log", "log"),
    ]
    try:
        for x_scale, y_scale in combinations:
            _, ax = plot_pair_length_histogram(
                dataset_path,
                max_bins=5,
                x_scale=x_scale,
                y_scale=y_scale,
                interactive_scale_toggle=False,
            )
            assert ax.get_xscale() == x_scale
            assert ax.get_yscale() == y_scale
    finally:
        dataset_path.unlink(missing_ok=True)

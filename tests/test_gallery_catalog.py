"""The selected bases and denser layouts preserve measurements and print type."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gallery.build import ALTERNATIVES, BASES
from gallery.data import absolute_config, load_absolute_data
from gallery import tables


def test_catalog_has_five_pairs_in_figure_order():
    assert BASES == ["1C-09-absolute-unit", "2D-compact", "3E-compact", "4A-compact", "5A-compact"]
    assert ALTERNATIVES == [key + "-dense" for key in BASES]
    assert len(set(BASES + ALTERNATIVES)) == 10
    assert [v["id"] for v in absolute_config()["variants"]] == [BASES[0], ALTERNATIVES[0]]


def test_dense_table_preserves_cells_fonts_colors_and_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "OUT", tmp_path)
    (tmp_path / "data").mkdir()
    frame = load_absolute_data()
    baseline, alternative = [tables.absolute_table(frame, v) for v in absolute_config()["variants"]]
    try:
        assert baseline.get_figwidth() == alternative.get_figwidth() == 6.2
        assert alternative.get_figheight() < baseline.get_figheight()
        first, second = baseline.axes[0], alternative.axes[0]
        assert len(first.texts) == len(second.texts) == 24 * 9
        assert [(t.get_text(), t.get_fontsize()) for t in first.texts] == [
            (t.get_text(), t.get_fontsize()) for t in second.texts
        ]
        np.testing.assert_array_equal(first.images[0].get_array(), second.images[0].get_array())
        for fig in (baseline, alternative):
            assert np.isclose(fig.axes[0].get_position().height * fig.get_figheight(), 5.28)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for text in fig.texts:
                box = text.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0
                assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
            caption_top = fig.texts[-1].get_window_extent(renderer).y1
            assert caption_top < fig.axes[0].get_window_extent(renderer).y0
        renderer = alternative.canvas.get_renderer()
        headers = alternative.axes[0].get_xticklabels()
        ticks = [t for ax in alternative.axes[1:] for t in ax.get_xticklabels()]
        assert max(t.get_window_extent(renderer).y1 for t in headers) < min(
            t.get_window_extent(renderer).y0 for t in ticks
        )
    finally:
        plt.close(baseline)
        plt.close(alternative)

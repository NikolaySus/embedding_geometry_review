"""Static density alternatives preserve the original compact measurements."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gallery.data import G, Q, load_data
from gallery.surfaces import compact_surface, large_surface
from gallery import trajectories


@pytest.fixture(autouse=True)
def print_style():
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8}):
        yield
    plt.close("all")


def pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def old_surface(values, tri, kind):
    # Frozen operations from build.py before extracting the compact interface.
    fig, labels = large_surface(values, tri, kind, -60, 6.5)
    for n, ax in enumerate(fig.axes):
        ax.set_position(
            [0.065 + (n % 2) * 0.49, 0.545 if n < 2 else 0.165, 0.445, 0.405]
        )
        x, y = ax.get_xlim(), ax.get_ylim()
        ax.set_box_aspect((x[1] - x[0], y[1] - y[0], 1.2), zoom=1.19)
    return fig, labels


@pytest.mark.parametrize("kind", ["2D", "3E", "5A"])
def test_surface_density(kind):
    _, mean, residual, tri = load_data()
    values = {"2D": mean[G], "3E": mean[Q[1:]], "5A": residual}[kind]
    old, _ = old_surface(values, tri, kind)
    base, base_labels = compact_surface(values, tri, kind)
    np.testing.assert_array_equal(pixels(base), pixels(old))
    dense, labels = compact_surface(values, tri, kind, dense=True)
    np.testing.assert_allclose(dense.get_size_inches(), [6.2, 6.65])
    assert len(labels) == len(base_labels) == 88
    caption = dense.texts[0].get_text()
    assert len(caption) < len(base.texts[0].get_text())
    for caveat in ["между ними измерений нет", "высота слоя условна",
                   "Мелкие подписи экспериментальны", "Разброс не показан"]:
        assert caveat in caption
    for before, after in zip(base.axes, dense.axes):
        assert sum(t.axes is after for t in labels) == 22
        assert (before.azim, before.elev) == (after.azim, after.elev)
        np.testing.assert_array_equal(before.get_proj(), after.get_proj())
        for getter in ["get_xlim", "get_ylim", "get_zlim", "get_box_aspect"]:
            np.testing.assert_array_equal(getattr(before, getter)(), getattr(after, getter)())
        np.testing.assert_allclose(
            before.get_position().size * base.get_size_inches(),
            after.get_position().size * dense.get_size_inches(),
        )
        for a, b in zip(before.texts, after.texts):
            assert a.get_text() == b.get_text()
            assert a.get_fontsize() == b.get_fontsize()
            np.testing.assert_array_equal(a.get_position_3d(), b.get_position_3d())
        for a, b in zip(before.collections, after.collections):
            if hasattr(a, "_offsets3d"):
                np.testing.assert_array_equal(a._offsets3d, b._offsets3d)
                assert (a.norm.vmin, a.norm.vmax) == (b.norm.vmin, b.norm.vmax)


def test_trajectory_density(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(trajectories, "OUT", tmp_path)
    old = trajectories.endpoints()
    old.subplots_adjust(left=.11, right=.99, top=.91, bottom=.20,
                        hspace=.30, wspace=.29)
    for ax in old.axes:
        ax.set_xlim(-5, 380)
    base = trajectories.compact_trajectories()
    np.testing.assert_array_equal(pixels(base), pixels(old))
    records = (tmp_path / "data/endpoints.csv").read_bytes()
    dense = trajectories.compact_trajectories(dense=True)
    assert records == (tmp_path / "data/endpoints.csv").read_bytes()
    np.testing.assert_allclose(dense.get_size_inches(), [6.2, 5.2])
    pixels(dense)
    renderer = dense.canvas.get_renderer()
    for before, after in zip(base.axes, dense.axes):
        assert before.get_xlim() == after.get_xlim()
        assert before.get_ylim() == after.get_ylim()
        for a, b in zip(before.lines, after.lines):
            np.testing.assert_array_equal(a.get_xydata(), b.get_xydata())
        for a, b in zip(before.collections, after.collections):
            for p, q in zip(a.get_paths(), b.get_paths()):
                np.testing.assert_array_equal(p.vertices, q.vertices)
        assert len(before.texts) == len(after.texts) == 4
        for a, b in zip(before.texts, after.texts):
            assert a.get_text() == b.get_text()
            assert a.get_fontsize() == b.get_fontsize()
            assert a.xy == b.xy
            assert a.get_position() == b.get_position()
            bbox = b.get_window_extent(renderer)
            assert dense.bbox.contains(bbox.x0, bbox.y0)
            assert dense.bbox.contains(bbox.x1, bbox.y1)

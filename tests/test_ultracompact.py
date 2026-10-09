"""Single-row layouts preserve all observations and opt-in document behavior."""

import matplotlib.pyplot as plt
from copy import copy
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pytest
from docx import Document

from gallery.bundles import load_bundle
from gallery.data import G, Q
from gallery.surfaces import compact_surface
from publication import document, ultracompact
from publication import unified_assets as shared


def test_front_matter_is_unchanged():
    compact = (shared.ROOT / "manuscript/compact.md").read_text()
    ultra = (shared.ROOT / "manuscript/ultracompact.md").read_text()
    assert ultra.split("## 1.")[0] == compact.split("## 1.")[0]


@pytest.mark.parametrize("kind", ["2D", "3E", "5A"])
def test_surface_row_preserves_labels_camera_limits_and_data(kind):
    bundle = load_bundle(shared.SETTINGS["experiment"], shared.SETTINGS["model"])
    frame = bundle["deltas"]
    mean = frame[frame.branch != "control"].groupby("branch")[Q + G].mean()
    values = {"2D": mean[G], "3E": mean[Q[1:]], "5A": bundle["residual"]}[kind]
    fig, labels = compact_surface(values, bundle["triangles"], kind, dense=True)
    assert len(labels) == 4 * 22
    before = [(ax.elev, ax.azim, ax.get_xlim(), ax.get_ylim(), ax.get_zlim(),
               [t.get_text() for t in ax.texts], [tuple(np.array(a) for a in l.get_data_3d()) for l in ax.lines]) for ax in fig.axes]
    ultracompact.surface_row(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, recorded in zip(fig.axes, before):
        assert (ax.elev, ax.azim, ax.get_xlim(), ax.get_ylim(), ax.get_zlim(), [t.get_text() for t in ax.texts]) == recorded[:6]
        for line, old in zip(ax.lines, recorded[6]):
            for actual, expected in zip(line.get_data_3d(), old):
                np.testing.assert_array_equal(actual, expected)
        for text in ax.texts:
            # Text3D restores world coordinates after draw; measure projected text.
            x, y, _ = proj3d.proj_transform(*text.get_position_3d(), ax.get_proj())
            projected = copy(text)
            projected.set_position((x, y))
            box = projected.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    assert np.ptp([ax.get_position().y0 for ax in fig.axes]) < 1e-9
    plt.close(fig)


def test_trajectory_row_preserves_means_and_sd():
    fig = shared.trajectory_figure()
    before = [[line.get_ydata().copy() for line in ax.lines] for ax in fig.axes]
    bands = [[path.vertices.copy() for collection in ax.collections for path in collection.get_paths()] for ax in fig.axes]
    ultracompact.trajectory_row(fig)
    for ax, means, old_bands in zip(fig.axes, before, bands):
        for line, mean in zip(ax.lines, means):
            np.testing.assert_array_equal(line.get_ydata(), mean)
        for path, old in zip([p for c in ax.collections for p in c.get_paths()], old_bands):
            np.testing.assert_array_equal(path.vertices, old)
    assert len(fig.legends) == 1 and len(fig.legends[0].texts) == 6
    plt.close(fig)


def test_references_page_break_is_opt_in(tmp_path):
    source, refs = tmp_path / "article.md", tmp_path / "refs.json"
    source.write_text("# Title\n\nText.\n\n## Список литературы\n\n{{REFERENCES}}\n")
    refs.write_text("[]")
    for enabled in (False, True):
        output = tmp_path / f"{enabled}.docx"
        document.build_docx(source, refs, output, references_new_page=enabled)
        doc = Document(output)
        heading = next(p for p in doc.paragraphs if p.text == "Список литературы")
        assert bool(heading.paragraph_format.page_break_before) == enabled
        assert doc.styles["Normal"].font.size.pt == 12

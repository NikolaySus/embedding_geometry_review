"""Gallery contracts use only the migrated package and curated source inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gallery.build import camera_figure
from gallery.data import ASSETS, ROOT, SRC, G, Q, load_data, layout_config
from gallery.interactive import apply_font, apply_roof, compact_scene, snap_camera


def test_package_root_and_source_integrity():
    assert ROOT == Path(__file__).resolve().parents[1]
    for name, digest in json.loads((ASSETS / "inputs.json").read_text()).items():
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == digest


def test_measured_data_contract():
    frame, mean, residual, triangles = load_data()
    assert len(frame) == 69
    assert mean.shape == (22, 9)
    assert residual.shape == (22, 4)
    assert triangles.shape == (27, 3)
    assert set(triangles.flat) == set(range(22))
    for branch in ["mix_s100_c000_r000", "mix_s000_c100_r000", "mix_s000_c000_r100"]:
        np.testing.assert_allclose(residual.loc[branch], 0, atol=1e-14)


@pytest.mark.parametrize("kind", ["2D", "3E", "5A"])
@pytest.mark.parametrize("panel", range(1, 5))
def test_scene_controls_without_generated_inputs(kind, panel):
    _, mean, residual, triangles = load_data()
    values = {"2D": mean[G], "3E": mean[Q[1:]], "5A": residual}[kind]
    cfg = layout_config()[kind]
    fig = compact_scene(
        values, triangles, kind, panel, cfg["azimuth"], cfg["label_font"]
    )
    assert fig.layout.title.text == f"{kind}-compact-{panel}"
    assert len(fig.data) == 28
    assert fig.layout.scene.camera.projection.type == "orthographic"
    assert fig.layout.scene.dragmode == "turntable"
    np.testing.assert_allclose(fig.data[0].z, values.iloc[:, panel - 1])
    for ratio in [0.15, 0.45, 1.5]:
        apply_roof(fig, ratio)
        np.testing.assert_allclose(
            fig.data[fig.layout.meta["roof_text_index"]].z,
            fig.layout.meta["high"] + ratio * fig.layout.meta["span"],
        )
    for size in [5, 6.5, 8, 12]:
        apply_font(fig, size)
        assert fig.data[
            fig.layout.meta["roof_text_index"]
        ].textfont.size == size * 1000 / (6.2 * 72)


@pytest.mark.parametrize("angle", range(0, 360, 10))
def test_camera_snap_preserves_radius_and_elevation(angle):
    camera, chosen = snap_camera(
        {
            "eye": {
                "x": 2 * np.cos(np.deg2rad(angle)),
                "y": 2 * np.sin(np.deg2rad(angle)),
                "z": 0.83,
            }
        }
    )
    assert chosen in range(0, 360, 60)
    assert camera["eye"]["z"] == 0.83
    assert np.isclose(np.hypot(camera["eye"]["x"], camera["eye"]["y"]), 2)
    assert camera["center"] == dict(x=0, y=0, z=0)
    assert camera["up"] == dict(x=0, y=0, z=1)


def test_sample_camera_without_build():
    state = json.loads((ASSETS / "examples/font8-az60.camera.json").read_text())
    fig = camera_figure(state)
    assert not fig.layout.updatemenus
    assert snap_camera(fig.layout.scene.camera.to_plotly_json())[1] == 60
    assert fig.data[fig.layout.meta["roof_text_index"]].textfont.size == 8 * 1000 / (
        6.2 * 72
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(variant="../outside"),
        dict(roof_ratio=0),
        dict(label_font=20),
        dict(label_font=float("nan")),
    ],
)
def test_invalid_camera_rejected(change):
    state = dict(variant="2D-compact-1", roof_ratio=0.45, label_font=6.5)
    state.update(change)
    with pytest.raises(ValueError):
        camera_figure(state)


@pytest.mark.parametrize("kind", ["2D", "3E", "5A"])
@pytest.mark.parametrize("panel", range(1, 5))
def test_camera_export_roundtrip(kind, panel):
    identity = f"{kind}-compact-{panel}"
    state = dict(
        variant=identity,
        roof_ratio=0.8,
        label_font=9,
        camera=snap_camera(dict(eye=dict(x=1, y=2, z=1.73)))[0],
        surface_visible=False,
    )
    figure = camera_figure(json.loads(json.dumps(state)))
    restored = dict(
        state,
        camera=figure.layout.scene.camera.to_plotly_json(),
        aspectratio=figure.layout.scene.aspectratio.to_plotly_json(),
        xaxis=figure.layout.scene.xaxis.to_plotly_json(),
        yaxis=figure.layout.scene.yaxis.to_plotly_json(),
        zaxis=figure.layout.scene.zaxis.to_plotly_json(),
    )
    again = camera_figure(json.loads(json.dumps(restored)))
    np.testing.assert_allclose(figure.data[0].z, again.data[0].z)
    np.testing.assert_allclose(figure.data[0].intensity, again.data[0].intensity)
    assert again.data[0].visible is False
    assert again.layout.scene.camera.eye.z == 1.73
    np.testing.assert_allclose(
        again.data[again.layout.meta["roof_text_index"]].z,
        again.layout.meta["high"] + 0.8 * again.layout.meta["span"],
    )
    assert again.data[
        again.layout.meta["roof_text_index"]
    ].textfont.size == 9 * 1000 / (6.2 * 72)

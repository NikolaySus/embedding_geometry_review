import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import plotly.io as pio
from pypdf import PdfReader
from .build import OUT, BASES, PALETTES, snap_camera, apply_font
from .data import ASSETS, SRC, Q, G, load_data


def main():
    for name, digest in json.loads((ASSETS / "inputs.json").read_text()).items():
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == digest, name
    _, vertices, residual, triangles = load_data()
    np.testing.assert_array_equal(
        pd.read_csv(OUT / "data/triangles.csv").to_numpy(), triangles
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(OUT / "data/residual_means.csv", index_col=0),
        residual,
        check_names=False,
    )
    data = pd.read_csv(OUT / "data/quality_geometry_by_seed.csv")
    assert (OUT / "data/quality_geometry_by_seed.csv").read_bytes() == (
        SRC / "quality_geometry_by_seed.csv"
    ).read_bytes()
    mean = data.groupby("branch")[Q + G].mean()
    for key, mode, palettes in PALETTES:
        s = pd.read_csv(OUT / "data" / f"{key}-scales.csv").set_index("column")
        for c in mean:
            v = mean[c]
            low, high = v.min(), v.max()
            if mode == "symmetric":
                low = -abs(v).max()
                high = -low
            elif mode == "zero":
                low = min(low, 0)
                high = max(high, 0)
            elif mode == "rank":
                low, high = 0, 1
            elif mode == "quantile":
                low, high = v.quantile([0.05, 0.95])
            assert np.isclose(s.loc[c, "lower"], low) and np.isclose(
                s.loc[c, "upper"], high
            )
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim=(-0.4, 1.2), ylim=(-0.5, 1.1), zlim=(-1, 1))
    ax.set_box_aspect((1.6, 1.6, 1.2))
    ax.set_proj_type("ortho")
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    for angle in range(0, 360, 60):
        for elevation in [20, 55, 80]:
            ax.view_init(elevation, angle)
            x, y, _ = proj3d.proj_transform(*xyz.T, ax.get_proj())
            assert min(abs(x[i] - x[(i + 1) % 3]) for i in range(3)) < 1e-10
    plt.close(fig)
    for angle in range(0, 360, 10):
        camera, chosen = snap_camera(
            dict(
                eye=dict(
                    x=np.cos(np.deg2rad(angle)), y=np.sin(np.deg2rad(angle)), z=0.83
                )
            )
        )
        assert chosen in range(0, 360, 60) and camera["eye"]["z"] == 0.83
    expected_panels = {
        f"{kind}-compact-{panel}"
        for kind in ["2D", "3E", "5A"]
        for panel in range(1, 5)
    }
    paths = list((OUT / "interactive").glob("*.figure.json"))
    assert {p.name.removesuffix(".figure.json") for p in paths} == expected_panels
    assert {p.stem for p in (OUT / "interactive").glob("*.html")} == expected_panels | {
        "index"
    }
    assert {
        p.name.removesuffix(".camera.json")
        for p in (OUT / "interactive").glob("*.camera.json")
    } == expected_panels
    for path in paths:
        f = pio.read_json(path)
        sx = f.layout.scene.xaxis.range
        sy = f.layout.scene.yaxis.range
        identity = path.name.removesuffix(".figure.json")
        kind, _, panel = identity.split("-")
        values = (
            {"2D": vertices[G], "3E": vertices[Q[1:]], "5A": residual}[kind]
            .iloc[:, int(panel) - 1]
            .to_numpy()
        )

        def array(value):
            if isinstance(value, dict) and "bdata" in value:
                import base64

                return np.frombuffer(
                    base64.b64decode(value["bdata"]), dtype=value["dtype"]
                )
            return np.asarray(value)

        np.testing.assert_allclose(array(f.data[0].z), values, atol=1e-15)
        np.testing.assert_array_equal(array(f.data[0].i), triangles[:, 0])
        assert f.layout.scene.dragmode == "turntable"
        assert f.layout.scene.camera.projection.type == "orthographic"
        state = json.loads(path.with_name(identity + ".camera.json").read_text())
        assert (
            state["variant"] == identity
            and state["roof_ratio"] == 0.45
            and state["label_font"] == 6.5
            and state["azimuth"] == 300
        )
        assert state["camera"] == f.layout.scene.camera.to_plotly_json()
        assert len(f.data[f.layout.meta["roof_text_index"]].text) == 22
        np.testing.assert_allclose(
            f.data[f.layout.meta["roof_text_index"]].z,
            f.layout.meta["high"] + 0.45 * f.layout.meta["span"],
        )
        assert np.isclose(
            f.layout.scene.aspectratio.x / (sx[1] - sx[0]),
            f.layout.scene.aspectratio.y / (sy[1] - sy[0]),
        )
        for size in [5, 6.5, 9, 12]:
            apply_font(f, size)
            assert np.isclose(
                f.data[f.layout.meta["roof_text_index"]].textfont.size,
                size * 1000 / (6.2 * 72),
            )
        html = path.with_name(path.name.replace(".figure.json", ".html")).read_text()
        assert (
            "label_font" in html
            and "azimuth" in html
            and "roof_ratio" in html
            and "plotly_relayout" in html
        )
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert len(manifest["variants"]) == 17
    expected_variants = (
        set(BASES)
        | {"1C-" + key for key, _, _ in PALETTES}
        | {"2D-compact", "3E-compact", "4A-compact", "5A-compact"}
    )
    assert {v["id"] for v in manifest["variants"]} == expected_variants
    assert {v["id"] for v in manifest["variants"] if v["base"]} == set(BASES)
    for identity in expected_variants:
        assert len(PdfReader(OUT / "figures" / f"{identity}.pdf").pages) == 1
        assert (OUT / "figures" / f"{identity}.png").stat().st_size > 10000
    assert len(PdfReader(OUT / "COMPARISON.pdf").pages) == 19
    result = dict(
        status="passed",
        variants=17,
        pages=19,
        palette_alternatives=8,
        interactive_panels=12,
        azimuths=list(range(0, 360, 60)),
        checks=[
            "palette limits",
            "unchanged source inputs",
            "isotropic XY scaling",
            "vertical edge projection at 18 views",
            "font units",
            "saved settings",
        ],
    )
    (OUT / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

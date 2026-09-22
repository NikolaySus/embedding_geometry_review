"""Build every curated gallery variant directly from compact source inputs."""

import argparse
import json
import shutil
import textwrap
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from matplotlib.backends.backend_pdf import PdfPages
from .data import OUT, SRC, Q, G, load_data, layout_config
from .tables import PALETTES, palette_table, independent_table
from .surfaces import large_surface
from .trajectories import endpoints
from .interactive import compact_scene, interactive, apply_roof, apply_font, snap_camera

BASES = ["1C-alternating", "2D-large", "3E-large", "4A-endpoints", "5A-large"]


def camera_figure(state):
    identity = state["variant"]
    valid = {
        f"{kind}-compact-{panel}"
        for kind in ["2D", "3E", "5A"]
        for panel in range(1, 5)
    }
    if identity not in valid:
        raise ValueError("Unknown variant")
    ratio = float(state.get("roof_ratio", 0.45))
    font = float(state.get("label_font", 6.5))
    if not 0.15 <= ratio <= 1.5 or not 5 <= font <= 12:
        raise ValueError("Roof or font outside supported range")
    kind, _, panel = identity.split("-")
    _, mean, residual, tri = load_data()
    values = {"2D": mean[G], "3E": mean[Q[1:]], "5A": residual}[kind]
    cfg = layout_config()[kind]
    f = compact_scene(values, tri, kind, int(panel), cfg["azimuth"], cfg["label_font"])
    f.update_layout(
        scene={
            k: state[k]
            for k in ["camera", "aspectratio", "xaxis", "yaxis", "zaxis"]
            if k in state
        }
    )
    f.layout.scene.camera, _ = snap_camera(f.layout.scene.camera.to_plotly_json())
    sx = f.layout.scene.xaxis.range
    sy = f.layout.scene.yaxis.range
    f.layout.scene.aspectratio.x = sx[1] - sx[0]
    f.layout.scene.aspectratio.y = sy[1] - sy[0]
    apply_roof(f, ratio)
    apply_font(f, font)
    f.layout.updatemenus = ()
    f.data[0].visible = state.get("surface_visible", True)
    return f


def export_camera(path):
    state = json.loads(Path(path).read_text())
    f = camera_figure(state)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        f.write_image(
            OUT / "figures" / f"{state['variant']}-camera.{ext}",
            width=1000,
            height=800,
            scale=2 if ext == "png" else 1,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera")
    args = p.parse_args()
    if args.camera:
        return export_camera(args.camera)
    for folder in ["data", "figures", "interactive", "renders"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8}
    )
    frame, m, residual, tri = load_data()
    config = layout_config()
    shutil.copyfile(
        SRC / "quality_geometry_by_seed.csv", OUT / "data/quality_geometry_by_seed.csv"
    )
    residual.to_csv(OUT / "data/residual_means.csv")
    pd.DataFrame(tri, columns=["i", "j", "k"]).to_csv(
        OUT / "data/triangles.csv", index=False
    )
    variants = []
    baselines = {
        "1C-alternating": independent_table(frame, True),
        "4A-endpoints": endpoints(),
    }
    for kind, values in [("2D", m[G]), ("3E", m[Q[1:]]), ("5A", residual)]:
        cfg = config[kind]
        baselines[kind + "-large"] = large_surface(
            values, tri, kind, cfg["azimuth"], cfg["label_font"]
        )[0]
    for key in BASES:
        fig = baselines[key]
        fig.savefig(OUT / "figures" / f"{key}.pdf")
        fig.savefig(OUT / "figures" / f"{key}.png", dpi=180)
        plt.close(fig)
        variants.append(
            dict(id=key, base=True, note="Выбранная основа, без изменений.")
        )

    def save(key, fig, note):
        fig.savefig(OUT / "figures" / f"{key}.pdf")
        fig.savefig(OUT / "figures" / f"{key}.png", dpi=180)
        plt.close(fig)
        variants.append(dict(id=key, base=False, note=note))

    for key, mode, palettes in PALETTES:
        save(
            "1C-" + key,
            palette_table(frame, key, mode, palettes),
            "Палитры: "
            + ", ".join(palettes)
            + "; нормировка: "
            + mode
            + ". Числа и SD одинаковы во всех вариантах.",
        )
    for kind, values in [("2D", m[G]), ("3E", m[Q[1:]]), ("5A", residual)]:
        fig, labels = large_surface(values, tri, kind, -60, 6.5)
        for n, ax in enumerate(fig.axes):
            ax.set_position(
                [0.065 + (n % 2) * 0.49, 0.545 if n < 2 else 0.165, 0.445, 0.405]
            )
            x = ax.get_xlim()
            y = ax.get_ylim()
            ax.set_box_aspect((x[1] - x[0], y[1] - y[0], 1.2), zoom=1.19)
        save(
            kind + "-compact",
            fig,
            "Уплотнённая сетка 2×2; одинаковый масштаб единиц X/Y. Азимут 300°: одна сторона основания вертикальна в проекции. Шрифты прежние.",
        )
        for panel in range(1, 5):
            interactive(
                values,
                tri,
                kind,
                panel,
                config[kind]["azimuth"],
                config[kind]["label_font"],
            )
    fig = endpoints()
    fig.subplots_adjust(
        left=0.11, right=0.99, top=0.91, bottom=0.20, hspace=0.30, wspace=0.29
    )
    for ax in fig.axes:
        ax.set_xlim(-5, 380)
    save(
        "4A-compact",
        fig,
        "Меньше межпанельные промежутки и правая область после шага 270; конечные числа сохранены.",
    )
    variants.sort(key=lambda v: (v["id"][0], not v["base"], v["id"]))
    writer = PdfWriter()
    with PdfPages(OUT / "renders/cover.pdf") as pdf:
        for start in [0, 10]:
            f = plt.figure(figsize=(8.27, 11.69))
            f.text(
                0.08,
                0.94,
                "Галерея: палитры, компактность, шесть азимутов",
                fontsize=13,
            )
            for i, v in enumerate(variants[start : start + 10]):
                f.text(
                    0.1,
                    0.85 - i * 0.065,
                    f"{start + i + 3:02d}   {v['id']}"
                    + (" (основа)" if v["base"] else ""),
                    fontsize=11,
                )
            f.text(
                0.08,
                0.07,
                "Цель: статические рисунки. HTML сохраняет высоту слоя,\nразмер чисел, дискретный азимут и свободную высоту обзора.",
                fontsize=9,
            )
            pdf.savefig(f)
            plt.close(f)
    writer.append(OUT / "renders/cover.pdf")
    for i, v in enumerate(variants):
        f = plt.figure(figsize=(8.27, 11.69))
        f.text(0.08, 0.955, v["id"], fontsize=13)
        f.text(0.08, 0.07, "\n".join(textwrap.wrap(v["note"], 90)), fontsize=9)
        f.text(0.08, 0.03, str(i + 3), fontsize=8)
        path = OUT / "renders" / f"{v['id']}.pdf"
        f.savefig(path)
        plt.close(f)
        page = PdfReader(path).pages[0]
        src = PdfReader(OUT / "figures" / f"{v['id']}.pdf").pages[0]
        page.merge_transformed_page(
            src,
            Transformation().translate(
                tx=74, ty=11.69 * 72 - 70 - float(src.mediabox.height)
            ),
        )
        writer.add_page(page)
    writer.write(OUT / "COMPARISON.pdf")
    (OUT / "manifest.json").write_text(
        json.dumps(dict(variants=variants), ensure_ascii=False, indent=2)
    )
    links = "".join(
        f'<li><a href="{p.name}">{p.stem}</a></li>'
        for p in sorted((OUT / "interactive").glob("*.html"))
        if p.name != "index.html"
    )
    (OUT / "interactive/index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Галерея рисунков</h1><p>Азимут: шесть значений. После отпускания мыши ракурс привязывается к ближайшему. Высота обзора свободная. Наклон камеры вокруг направления взгляда отключён.</p><ul>'
        + links
        + "</ul>"
    )
    print(OUT / "COMPARISON.pdf")

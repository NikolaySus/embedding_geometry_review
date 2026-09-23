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
from .data import OUT, SRC, Q, G, load_data, layout_config, load_absolute_data, absolute_config
from .tables import PALETTES, absolute_table
from .surfaces import compact_surface
from .trajectories import compact_trajectories
from .interactive import compact_scene, interactive, apply_roof, apply_font, snap_camera

BASES = ["1C-09-absolute-unit", "2D-compact", "3E-compact", "4A-compact", "5A-compact"]
ALTERNATIVES = [identity + "-dense" for identity in BASES]


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
    _, m, residual, tri = load_data()
    config = layout_config()
    shutil.copyfile(
        SRC / "quality_geometry_by_seed.csv", OUT / "data/quality_geometry_by_seed.csv"
    )
    residual.to_csv(OUT / "data/residual_means.csv")
    pd.DataFrame(tri, columns=["i", "j", "k"]).to_csv(
        OUT / "data/triangles.csv", index=False
    )
    variants = []
    absolute = load_absolute_data()
    absolute.to_csv(OUT / "data/absolute_quality_geometry_by_seed.csv", index=False)
    absolute.groupby("branch")[Q + G].agg(["mean", "std"]).to_csv(OUT / "data/absolute_summary.csv")

    def save(key, fig, note):
        width, height = fig.get_size_inches()
        fig.savefig(OUT / "figures" / f"{key}.pdf")
        fig.savefig(OUT / "figures" / f"{key}.png", dpi=180)
        plt.close(fig)
        variants.append(
            dict(id=key, base=key in BASES, note=note,
                 width_inches=float(width), height_inches=float(height))
        )

    for variant in absolute_config()["variants"]:
        save(
            variant["id"], absolute_table(absolute, variant),
            "Абсолютные значения и M0; общая шкала качества 0–1 и отдельные шкалы геометрии. "
            + ("Плотная альтернатива: сокращены поля и подпись; размеры ячеек и шрифтов прежние."
               if variant.get("layout") == "dense" else "Выбранная основа, без изменений."),
        )

    for kind, values in [("2D", m[G]), ("3E", m[Q[1:]]), ("5A", residual)]:
        for dense in (False, True):
            fig, _ = compact_surface(values, tri, kind, dense=dense)
            save(
                kind + "-compact" + ("-dense" if dense else ""), fig,
                "Сетка 2×2; измеренные точки и шкалы сохранены. "
                + ("Плотная альтернатива: сокращены поля, межпанельное пространство и подпись; шрифты не уменьшены."
                   if dense else "Прежний compact теперь основа, без изменений."),
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
    for dense in (False, True):
        save(
            "4A-compact" + ("-dense" if dense else ""),
            compact_trajectories(dense=dense),
            "Траектории первого этапа; все шаги, конечные значения и SD сохранены. "
            + ("Плотная альтернатива: меньше промежутки, поля и высота рисунка; шрифты не уменьшены."
               if dense else "Прежний compact теперь основа, без изменений."),
        )
    variants.sort(key=lambda v: (v["id"][0], not v["base"], v["id"]))
    for variant in variants:
        base = next(v for v in variants if v["base"] and v["id"][0] == variant["id"][0])
        variant["area_reduction_percent"] = 100 * (
            1 - variant["width_inches"] * variant["height_inches"]
            / (base["width_inches"] * base["height_inches"])
        )
    pd.DataFrame(variants).to_csv(OUT / "data/layout_comparison.csv", index=False)
    cover_pages = (len(variants) + 9) // 10
    writer = PdfWriter()
    with PdfPages(OUT / "renders/cover.pdf") as pdf:
        for start in range(0, len(variants), 10):
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
                    f"{start + i + cover_pages + 1:02d}   {v['id']}"
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
        f.text(
            0.08, 0.922,
            f"{v['width_inches'] * 25.4:.1f} × {v['height_inches'] * 25.4:.1f} мм; "
            + ("основа" if v["base"] else f"площадь меньше на {v['area_reduction_percent']:.1f}%")
            + "; показано в исходном масштабе",
            fontsize=9,
        )
        f.text(0.08, 0.07, "\n".join(textwrap.wrap(v["note"], 90)), fontsize=9)
        f.text(0.08, 0.03, str(i + cover_pages + 1), fontsize=8)
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
    static_links = "".join(
        f'<li>{v["id"]} ({"основа" if v["base"] else "плотная альтернатива"}): '
        f'<a href="../figures/{v["id"]}.pdf">PDF</a> · '
        f'<a href="../figures/{v["id"]}.png">PNG</a>'
        + (f' · −{v["area_reduction_percent"]:.1f}% площади' if not v["base"] else '')
        + '</li>'
        for v in variants
    )
    (OUT / "interactive/index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Галерея рисунков</h1>'
        '<p><a href="../COMPARISON.pdf">Все варианты в PDF</a></p>'
        '<h2>Основы и плотные альтернативы</h2><ul>' + static_links + '</ul>'
        '<h2>Интерактивные панели</h2><p>Азимут: шесть значений. После отпускания мыши ракурс привязывается к ближайшему. Высота обзора свободная. Наклон камеры вокруг направления взгляда отключён.</p><ul>'
        + links
        + "</ul>", encoding="utf-8",
    )
    # Remove only named retired gallery products, not manually exported cameras.
    selected = set(BASES + ALTERNATIVES)
    retired = {"2D-large", "3E-large", "4A-endpoints", "5A-large", "1C-alternating"}
    retired.update("1C-" + key for key, _, _ in PALETTES)
    for folder in ["figures", "renders"]:
        directory = (OUT / folder).resolve()
        for identity in retired - selected:
            for extension in ("pdf", "png"):
                (directory / f"{identity}.{extension}").unlink(missing_ok=True)
    for name in ["scales_alternating.csv", "scales_independent.csv"] + [f"{key}-scales.csv" for key, _, _ in PALETTES]:
        (OUT / "data" / name).unlink(missing_ok=True)
    print(OUT / "COMPARISON.pdf")

"""Print-size tables with independent metric scales and palette alternatives."""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from .data import OUT, Q, G, LABEL, short, absolute_config

PALETTES = [
    ("01-symmetric-uniform", "symmetric", ["RdBu"]),
    ("02-symmetric-triple", "symmetric", ["RdBu", "PuOr", "PiYG"]),
    ("03-asymmetric-zero", "zero", ["RdBu", "PuOr"]),
    ("04-minmax-uniform", "minmax", ["viridis"]),
    ("05-minmax-alternating", "minmax", ["viridis", "cividis"]),
    ("06-minmax-warm-cool", "minmax", ["Blues", "YlOrBr"]),
    ("07-robust-quantiles", "quantile", ["viridis", "cividis"]),
    ("08-empirical-rank", "rank", ["viridis", "cividis"]),
]


def absolute_scales(mean, variant):
    config = absolute_config()
    bounds = variant["quality_range"]
    if bounds == "observed":
        bounds = [float(mean[Q].to_numpy().min()), float(mean[Q].to_numpy().max())]
    rows = []
    for c in Q + G:
        low, high = bounds if c in Q else config["geometry"][c]["range"]
        palette = config["quality_palette"] if c in Q else config["geometry"][c]["palette"]
        clipped = int(((mean[c] < low) | (mean[c] > high)).sum())
        if clipped:
            raise ValueError(f"Absolute scale would clip {c}: {low}..{high}")
        rows.append(dict(column=c, palette=palette, lower=low, upper=high, clipped=clipped))
    return pd.DataFrame(rows).set_index("column")


def absolute_table(frame, variant, *, data_dir=None):
    """Same numeric-table design as 1C-01, with explicit absolute scales."""
    mean = frame.groupby("branch")[Q + G].mean()
    order = ["m0"] + [b for b in mean.index if b != "m0"]
    mean = mean.loc[order]
    sd = frame.groupby("branch")[Q + G].std().loc[order]
    scales = absolute_scales(mean, variant)
    # Keep cell sizes and font sizes; reclaim vertical space above the table.
    dense = variant.get("layout") == "dense"
    height = 7.1 if dense else 7.6
    bottom = 1.06 if dense else 1.52
    fig = plt.figure(figsize=(6.2, height))
    ax = fig.add_axes([0.15, bottom / height, 0.83, 5.28 / height])
    rgba = np.empty((len(mean), 9, 4))
    for j, c in enumerate(Q + G):
        s = scales.loc[c]
        cmap = plt.colormaps[s.palette]
        colors = cmap(Normalize(s.lower, s.upper)(mean[c].to_numpy()))
        colors[:, :3] = 0.5 * colors[:, :3] + 0.5
        rgba[:, j] = colors
        for i, branch in enumerate(order):
            digits = 1 if c == "effective_rank" else 3
            ax.text(j, i, f"{mean.loc[branch,c]:.{digits}f}\n±{sd.loc[branch,c]:.{digits}f}",
                    ha="center", va="center", fontsize=7.5, linespacing=1.05)
    ax.imshow(rgba, aspect="auto")
    ax.set(xticks=range(9), xticklabels=[LABEL[c] + (" ↑" if c in Q else "") for c in Q + G],
           yticks=range(len(order)), yticklabels=["M0" if b == "m0" else short(b) for b in order])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=8)
    ax.set_xticks(np.arange(-0.5, 9), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.axvline(4.5, color="#555555", lw=1)
    ax.axhline(0.5, color="#555555", lw=0.8)
    heading_y = 6.96 if dense else 7.41
    fig.text(0.15, heading_y / height, "Абсолютное качество", fontsize=9)
    fig.text(0.63, heading_y / height, "Геометрия: отдельные шкалы", fontsize=8)
    # Legends use the very same lightened palettes as the cells.
    for c, x, width, title in [(Q[0], .15, .44, "Общая шкала AG / STS / Cls-tr / Clust / Ret")] + [
        (c, .15 + (5+j)*.83/9 + .008, .075, LABEL[c]) for j,c in enumerate(G)
    ]:
        s = scales.loc[c]
        bar = fig.add_axes([x, (6.68 if dense else 7.10) / height, width, .104 / height])
        colors = plt.colormaps[s.palette](np.linspace(0, 1, 256))
        colors[:, :3] = .5 * colors[:, :3] + .5
        bar.imshow(colors[None, :, :], aspect="auto", extent=[s.lower, s.upper, 0, 1])
        bar.set(yticks=[], xticks=[s.lower, s.upper])
        bar.set_xticklabels([f"{s.lower:.3g}", f"{s.upper:.3g}"])
        bar.get_xticklabels()[0].set_ha("left")
        bar.get_xticklabels()[-1].set_ha("right")
        bar.tick_params(length=2, labelsize=6.5, pad=2)
        bar.set_title(title, fontsize=6.5, pad=4)
        for spine in bar.spines.values():
            spine.set_visible(False)
    caption = (
        "Абсолютное среднее ± SD трёх seed; строки: веса STS:Cls:Ret.\n"
        "AG / Cls-tr: accuracy (Cls-tr без AG News); STS: Spearman.\n"
        "Clust: V-measure; Ret: nDCG@10; семейства усреднены по задачам.\n"
        "Красный → синий: ниже → выше, не Δ; равные числа ≠ равная полезность.\n"
        "Rank: эффективный ранг; Top10: доля дисперсии; kNN: соседи M0.\n"
        "Margin: query-document зазор. Для геометрии нет общего «лучше».\n"
        "M0: исходная модель; SD повторной оценки, не независимого обучения."
        if dense else
        "Ячейка: абсолютное среднее ± SD трёх seed; веса STS:Cls:Ret.\n"
        "AG: accuracy AG News; STS: Spearman; Cls-tr: accuracy без AG News;\n"
        "Clust: V-measure; Ret: nDCG@10. Семейства: среднее по задачам.\n"
        "Качество: красный — ниже, синий — выше; середина не означает Δ = 0.\n"
        "Одинаковый цвет качества означает одинаковое число, не равную полезность.\n"
        "Rank: эффективный ранг; Top10: доля дисперсии; kNN: доля соседей M0;\n"
        "Margin: query-document зазор. Геометрия не имеет общего «лучше/хуже».\n"
        "M0: исходная модель; её SD описывает повторную оценку, не обучение."
    )
    fig.text(.15, (.10 if dense else .216) / height, caption,
        fontsize=7, linespacing=1.3)
    destination = OUT / "data" if data_dir is None else data_dir
    scales.to_csv(destination / f"{variant['id']}-scales.csv")
    return fig


def heat_table(frame, split=False):
    mean = frame.groupby("branch")[Q + G].mean()
    sd = frame.groupby("branch")[Q + G].std()
    fig = plt.figure(figsize=(6.2, 8.0))
    groups = [G if split == "geometry" else Q] if split else [Q + G]
    scales = mean.abs().max()
    scales.loc[Q] = mean[Q].abs().to_numpy().max()
    for k, cols in enumerate(groups):
        ax = fig.add_axes([0.15, 0.14, 0.83, 0.78])
        rgba = plt.colormaps["RdBu"](
            Normalize(-1, 1)(mean[cols].to_numpy() / scales[cols].to_numpy())
        )
        rgba[:, :, :3] = 0.55 * rgba[:, :, :3] + 0.45
        ax.imshow(rgba, aspect="auto")
        for i, b in enumerate(mean.index):
            for j, c in enumerate(cols):
                digits = 1 if c == "effective_rank" else 3
                val = f"{mean.loc[b, c]:+.{digits}f}"
                err = f"±{sd.loc[b, c]:.{digits}f}"
                ax.text(
                    j,
                    i,
                    val + "\n" + err,
                    ha="center",
                    va="center",
                    fontsize=8,
                    linespacing=1.05,
                )
        ax.set(
            xticks=range(len(cols)),
            xticklabels=[LABEL[c] for c in cols],
            yticks=range(len(mean)),
            yticklabels=[short(b) for b in mean.index],
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0, labelsize=8)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(mean)), minor=True)
        ax.grid(which="minor", color="white", lw=0.6)
        ax.tick_params(which="minor", length=0)
    fig.text(
        0.15,
        0.04,
        "В ячейке: средняя Δ и ± SD (3 seed). Цвет: знак и величина Δ.\nКачество: общая шкала; геометрия: отдельная шкала столбца.\nКрасный: минус; синий: плюс. Геометрия: не оценка «лучше/хуже».",
        fontsize=8,
    )
    return fig


def range_table(frame, mode):
    fig = heat_table(frame)
    ax = fig.axes[0]
    m = frame.groupby("branch")[Q + G].mean()
    scales = m.abs().max()
    scales.loc[Q] = scales[Q].max()
    headers = []
    for c in m:
        precision = 1 if c == "effective_rank" else 3
        title = LABEL[c] + (" ↑" if c in Q else "")
        if mode in ["scale", "both"]:
            title += f"\n±{scales[c]:.{precision}f}"
        if mode in ["range", "both"]:
            title += f"\n{m[c].min():.{precision}f}\n…{m[c].max():.{precision}f}"
        headers.append(title)
    ax.set_xticklabels(headers, fontsize=8)
    ax.set_position([0.15, 0.14, 0.83, 0.69 if mode == "both" else 0.72])
    if mode == "range":
        fig.text(
            0.15,
            0.93,
            "Цвет: качество ±"
            + f"{scales[Q[0]]:.3f}"
            + "; геометрия: "
            + ", ".join(f"{LABEL[c]} ±{scales[c]:.3g}" for c in G),
            fontsize=8,
        )
    fig.text(
        0.15,
        0.095,
        "Δ относительно M0; ↑ больше лучше; STS:Cls:Ret.\nЗаголовок: ± предел цвета; min…max средних, не отдельных seed.",
        fontsize=8,
    )
    return fig


def independent_table(frame, alternate=False):
    fig = range_table(frame, "range")
    ax = fig.axes[0]
    m = frame.groupby("branch")[Q + G].mean()
    limits = m.abs().max()
    rgba = np.empty((len(m), len(m.columns), 4))
    names = []
    for j, c in enumerate(m):
        cmap = "PuOr" if alternate and j % 2 else "RdBu"
        names.append(cmap)
        rgba[:, j] = plt.colormaps[cmap](
            Normalize(-limits[c], limits[c])(m[c].to_numpy())
        )
    rgba[:, :, :3] = rgba[:, :, :3] * 0.55 + 0.45
    ax.images[0].set_data(rgba)
    for t in list(fig.texts):
        t.remove()
    labels = []
    for c in m:
        digits = 1 if c == "effective_rank" else 3
        labels.append(
            LABEL[c]
            + (" ↑" if c in Q else "")
            + f"\n±{limits[c]:.{digits}f}\n{m[c].min():.{digits}f}\n…{m[c].max():.{digits}f}"
        )
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_position([0.15, 0.15, 0.83, 0.69])
    fig.text(
        0.15,
        0.035,
        "Каждый столбец: собственная симметричная шкала ± max|средняя Δ|.\nЗаголовок: предел цвета и min…max средних; ↑ больше лучше.\nЯчейка: средняя Δ относительно M0 и ± SD трёх seed; STS:Cls:Ret.\n"
        + (
            "Палитры чередуются: знак читается по числу; интенсивность — по модулю."
            if alternate
            else "Красный: минус; синий: плюс. Цвет не сравним по величине между столбцами."
        )
        + "\nГеометрия не имеет универсального направления улучшения.",
        fontsize=8,
    )
    pd.DataFrame(
        {"column": m.columns, "limit": limits.to_numpy(), "palette": names}
    ).to_csv(
        OUT
        / "data"
        / ("scales_alternating.csv" if alternate else "scales_independent.csv"),
        index=False,
    )
    return fig


def palette_table(frame, key, mode, palettes):
    f = range_table(frame, "range")
    ax = f.axes[0]
    m = frame.groupby("branch")[Q + G].mean()
    rgba = np.empty((len(m), 9, 4))
    rows = []
    headers = []
    for j, c in enumerate(m):
        v = m[c]
        lo = float(v.min())
        hi = float(v.max())
        clipped = 0
        if mode == "symmetric":
            lo = -max(abs(lo), abs(hi))
            hi = -lo
            norm = Normalize(lo, hi)
            u = norm(v)
        elif mode == "zero":
            # One-sided columns use only their matching half of the diverging map.
            lo = min(lo, 0)
            hi = max(hi, 0)
            u = np.where(
                v < 0,
                0.5 + 0.5 * v / max(abs(lo), 1e-12),
                0.5 + 0.5 * v / max(hi, 1e-12),
            )
        elif mode == "rank":
            lo = 0.0
            hi = 1.0
            u = (v.rank(method="average") - 1) / (len(v) - 1)
        else:
            if mode == "quantile":
                lo, hi = v.quantile([0.05, 0.95])
                clipped = int(((v < lo) | (v > hi)).sum())
            u = Normalize(lo, hi, clip=True)(v)
        cmap = palettes[j % len(palettes)]
        colors = plt.colormaps[cmap](u)
        colors[:, :3] = colors[:, :3] * 0.5 + 0.5
        rgba[:, j] = colors
        digits = 1 if c == "effective_rank" else 3
        headers.append(
            LABEL[c] + (" ↑" if c in Q else "") + f"\n{lo:.{digits}f}\n…{hi:.{digits}f}"
        )
        rows.append(
            dict(column=c, mode=mode, palette=cmap, lower=lo, upper=hi, clipped=clipped)
        )
    ax.images[0].set_data(rgba)
    ax.set_xticklabels(headers, fontsize=8)
    for t in list(f.texts):
        t.remove()
    descriptions = {
        "symmetric": "Цвет: отдельный ±max|Δ|; середина палитры означает ноль.",
        "zero": "Цвет: min…0…max, по половине палитры на знак; шкала кусочно-линейная.",
        "minmax": "Цвет: от min к max каждого столбца; цвет НЕ кодирует знак относительно нуля.",
        "quantile": "Цвет: 5–95-й процентили; крайние цвета насыщаются, числа не обрезаны.",
        "rank": "Цвет: эмпирический ранг среднего внутри столбца, не величина изменения.",
    }
    f.text(
        0.15,
        0.04,
        descriptions[mode]
        + "\nЗаголовок: пределы цвета; ячейка: средняя Δ и ± SD трёх seed.\n↑ больше лучше; STS:Cls:Ret. Геометрия не имеет общего направления улучшения.\nШкалы столбцов независимы. Сравнивать величину по цвету между ними нельзя.\n"
        + (
            "В квантильном варианте крайние 5% с каждой стороны отмечены пределами шкалы."
            if mode == "quantile"
            else "При рангах 0…1 не являются диапазоном метрики."
            if mode == "rank"
            else "Точные значения неизменны; знак определяется числом."
        ),
        fontsize=8,
    )
    pd.DataFrame(rows).to_csv(OUT / "data" / f"{key}-scales.csv", index=False)
    return f

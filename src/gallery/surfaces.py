"""Measured-vertex surfaces and their print layout."""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from .data import LABEL, xy, ternary_grid, project


def surface(m, tri, kind, style, azim=-55, font=7):
    roof = "roof" in style
    gradient = "gradient" in style
    fig = plt.figure(figsize=(6.2, 7.4))
    labels = []
    x, y = xy(m.index)
    for n, c in enumerate(m):
        ax = fig.add_subplot(2, 2, n + 1, projection="3d")
        z = m[c].to_numpy()
        limit = np.abs(m.to_numpy()).max() if kind == "5A" else abs(z).max()
        norm = Normalize(-limit, limit)
        cmap = plt.colormaps["RdBu"]
        low = min(z.min(), 0)
        high = max(z.max(), 0)
        span = max(high - low, 1e-6)
        floor = low - 0.13 * span
        top = high + (0.45 if roof else 0.07) * span
        xyz = np.column_stack([x, y, z])
        ax.add_collection3d(
            Poly3DCollection(
                xyz[tri],
                facecolors=cmap(norm(z[tri].mean(axis=1))),
                edgecolors="#777",
                linewidths=0.35,
                alpha=0.6,
            )
        )
        sc = ax.scatter(x, y, z, c=z, cmap=cmap, norm=norm, s=12, depthshade=False)
        for a, b in ternary_grid():
            ends = np.array([project(a), project(b)])
            ax.plot(ends[:, 0], ends[:, 1], [floor] * 2, color="#aaa", lw=0.35)
        for w, name in [
            (np.array([1, 0, 0]), "STS"),
            (np.array([0, 1, 0]), "Cls"),
            (np.array([0, 0, 1]), "Ret"),
        ]:
            xx, yy = project(w)
            ax.text(xx, yy, floor, name, fontsize=8)
        if roof:
            for xx, yy, v in zip(x, y, z):
                ax.plot(
                    [xx, xx], [yy, yy], [floor, top], color="#666", alpha=0.22, lw=0.45
                )
                labels.append(
                    ax.text(
                        xx,
                        yy,
                        top,
                        f"{v:+.1f}" if c == "effective_rank" else f"{v:+.3f}",
                        fontsize=font,
                        ha="center",
                        va="center",
                    )
                )
        title = ("R " if kind == "5A" else "Δ ") + LABEL[c]
        ax.set(xticks=[], yticks=[], zlim=(floor, top + 0.09 * span))
        ax.grid(False)
        ax.view_init(55 if roof else 30, azim)
        ax.set_proj_type("ortho")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_alpha(0)
        ax.xaxis.line.set_color("none")
        ax.yaxis.line.set_color("none")
        if gradient:
            # A colored vertical ruler occupies the same role as the z axis.
            ax.set_zticks([])
            ax.zaxis.line.set_color("none")
            ticks = np.linspace(low, high, 4)
            gx = 0.5 + 0.9 * np.sin(np.deg2rad(azim))
            gy = 0.29 - 0.9 * np.cos(np.deg2rad(azim))
            zz = np.linspace(low, high, 100)
            segs = [[(gx, gy, a), (gx, gy, b)] for a, b in zip(zz[:-1], zz[1:])]
            ax.add_collection3d(
                Line3DCollection(
                    segs, colors=cmap(norm((zz[:-1] + zz[1:]) / 2)), linewidths=3
                )
            )
            for v in ticks:
                ax.text(
                    gx,
                    gy,
                    v,
                    f"{v:.1f}" if c == "effective_rank" else f"{v:.2f}",
                    fontsize=8,
                    ha="right",
                )
            ax.text(gx, gy, high + 0.20 * span, title, fontsize=8, ha="center")
            ax.set_xlim(min(-0.08, gx - 0.10), max(1.08, gx + 0.10))
            ax.set_ylim(min(-0.08, gy - 0.10), max(0.95, gy + 0.10))
        else:
            ax.set_zlabel(title, fontsize=8, labelpad=3)
            ax.tick_params(labelsize=8, pad=0)
            from matplotlib.ticker import MaxNLocator

            ax.zaxis.set_major_locator(MaxNLocator(4))
            fig.colorbar(
                sc,
                ax=ax,
                orientation="horizontal",
                shrink=0.65,
                fraction=0.04,
                pad=0.05,
            )
    fig.subplots_adjust(
        left=0.01, right=0.95, top=0.94, bottom=0.20, hspace=0.35, wspace=0.13
    )
    note = (
        "R = Δ смеси − сумма (доля × Δ чистой цели); R > 0 не означает выигрыш у M0."
        if kind == "5A"
        else "Δ относительно M0; цвет и высота означают одно свойство."
    )
    definitions = (
        "Rank: эффективный ранг; Top10: доля дисперсии первых 10 компонент.\nkNN: сохранение соседств; Margin: поисковый зазор."
        if kind == "2D"
        else "STS: близость; Cls-tr: перенос классификации; Clust: кластеризация; Ret: поиск."
    )
    fig.text(
        0.03,
        0.025,
        definitions
        + "\n"
        + note
        + "\n22 смеси; сетка долей 20%; синий: плюс, красный: минус.\nГрани соединяют измерения; промежуточные значения не измерены.\n"
        + (
            "Числа наверху: значения точек; высота верхнего слоя условна.\nМелкие подписи экспериментальны. Разброс не показан."
            if roof
            else "Разброс не показан; знак геометрии не определяет полезность."
            if kind == "2D"
            else "Разброс не показан."
        ),
        fontsize=8,
    )
    return fig, labels


def large_surface(m, tri, kind, azim, font):
    fig, labels = surface(m, tri, kind, "roof-gradient", azim, font)
    for n, ax in enumerate(fig.axes):
        ax.set_position([0.06 + (n % 2) * 0.49, 0.56 if n < 2 else 0.18, 0.445, 0.39])
        ax.set_box_aspect((1, 0.866, 1), zoom=1.17)
        c = m.columns[n]
        z = m[c].to_numpy()
        low = min(0, z.min())
        high = max(0, z.max())
        span = max(high - low, 1e-6)
        gx = 0.5 + 0.9 * np.sin(np.deg2rad(azim))
        gy = 0.29 - 0.9 * np.cos(np.deg2rad(azim))
        dx = 0.065 * np.sin(np.deg2rad(azim))
        dy = -0.065 * np.cos(np.deg2rad(azim))
        # Last five texts are the ruler's four ticks and metric label.
        ticktexts = list(ax.texts)[-5:-1]
        title = ax.texts[-1]
        for t, v in zip(ticktexts, np.linspace(low, high, 4)):
            t.set_position_3d((gx + 1.8 * dx, gy + 1.8 * dy, v))
            ax.plot([gx, gx + dx], [gy, gy + dy], [v, v], color="#333", lw=0.65)
        title.set_position_3d((gx, gy, high + 0.32 * span))
        for t in labels:
            if t.axes is ax:
                t.set_verticalalignment("bottom")
    return fig, labels

"""Training trajectories with endpoint means and seed uncertainty."""

import matplotlib.pyplot as plt
import pandas as pd
from .data import SRC, OUT, COLORS, LABEL


def endpoints(*, data_dir=None):
    d = pd.read_csv(SRC / "geometry_trajectories.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.8))
    records = []
    for ax, c in zip(
        axes.flat,
        [
            "effective_rank",
            "top10_eigenvalue_share",
            "mean_cosine",
            "neighborhood_preservation",
        ],
    ):
        values = []
        for (b, color), style in zip(COLORS.items(), ["-", "--", "-.", ":"]):
            s = d[d.branch == b].groupby("step")[c].agg(["mean", "std"])
            ax.plot(
                s.index, s["mean"], marker="o", ms=3, ls=style, color=color, label=b
            )
            ax.fill_between(
                s.index,
                s["mean"] - s["std"],
                s["mean"] + s["std"],
                color=color,
                alpha=0.13,
            )
            values.append((b, color, s["mean"].iloc[-1], s["std"].iloc[-1]))
        ax.set(
            xlim=(-5, 425),
            xticks=[0, 68, 135, 270],
            xlabel="Шаг обучения",
            ylabel=LABEL.get(c, "Cosine"),
        )
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.06)
        values.sort(key=lambda a: a[2])
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.18
        placed = []
        for _, _, v, _ in values:
            placed.append(max(v, placed[-1] + gap) if placed else max(v, lo + gap / 2))
        placed[-1] = min(placed[-1], hi - gap / 2)
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - gap)
        for (b, color, v, sd), y in zip(values, placed):
            digits = 1 if c == "effective_rank" else 3
            ax.annotate(
                f"{v:.{digits}f}\n±{sd:.{digits}f}",
                xy=(270, v),
                xytext=(292, y),
                fontsize=8,
                color=color,
                va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=color),
                annotation_clip=False,
            )
            records.append(dict(metric=c, branch=b, step=270, mean=v, sd=sd, label_y=y))
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.9, bottom=0.20, hspace=0.5, wspace=0.4
    )
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        fontsize=8,
    )
    fig.text(
        0.04,
        0.04,
        "Первый этап: отдельные режимы, не смеси; три seed.\nМаркеры: шаги 0, 68, 135, 270; линии лишь соединяют измерения.\nПолоса: ± SD; у конца линии: среднее и ± SD на шаге 270.\nПодписи разведены по высоте и соединены с фактическими конечными точками.",
        fontsize=8,
    )
    destination = OUT / "data" if data_dir is None else data_dir
    pd.DataFrame(records).to_csv(destination / "endpoints.csv", index=False)
    return fig


def compact_trajectories(dense=False, *, data_dir=None):
    """Return compact endpoint means/SD, optionally on a shorter print canvas."""
    fig = endpoints(data_dir=data_dir)
    fig.subplots_adjust(
        left=0.11, right=0.99, top=0.91, bottom=0.20, hspace=0.30, wspace=0.29
    )
    for ax in fig.axes:
        ax.set_xlim(-5, 380)
    if dense:
        fig.set_size_inches(6.2, 5.2)
        fig.subplots_adjust(
            left=0.11, right=0.99, top=0.91, bottom=0.22, hspace=0.29, wspace=0.27
        )
        for text in fig.texts:
            text.set_text(
                text.get_text()
                .replace("Первый этап:", "Этап 1:")
                .replace("у конца линии:", "в конце:")
                .replace("Подписи разведены по высоте и соединены с фактическими конечными точками.",
                         "Подписи разведены по высоте; выноски ведут к фактическим концам линий.")
            )
    return fig

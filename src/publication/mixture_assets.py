from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "pilot_v2_mixtures"
ASSET_DIR = ROOT / "build" / "full" / "mixtures"
DATA_DIR = ASSET_DIR / "data"
FIGURE_DIR = ASSET_DIR / "figures"

OBJECTIVES = ("sts", "classification", "retrieval")
BROAD_FAMILIES = ("sts", "classification_transfer", "clustering", "retrieval")
FAMILY_LABELS = {
    "sts": "STS",
    "classification": "Классификация, включая AG News",
    "classification_transfer": "Перенос классификации",
    "clustering": "Кластеризация",
    "retrieval": "Поиск",
}
COLORS = {
    "sts": "#2878B5",
    "classification": "#C54E52",
    "retrieval": "#2A9D6F",
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def weights_from_branch(branch: str) -> dict[str, float]:
    if branch == "control":
        return {"sts": math.nan, "classification": math.nan, "retrieval": math.nan}
    if branch == "mix_s033_c033_r033":
        return {objective: 1 / 3 for objective in OBJECTIVES}
    match = re.fullmatch(r"mix_s(\d{3})_c(\d{3})_r(\d{3})", branch)
    if not match:
        raise ValueError(f"Cannot parse branch: {branch}")
    return dict(zip(OBJECTIVES, (int(value) / 100 for value in match.groups()), strict=True))


def nondominated(frame: pd.DataFrame, columns: tuple[str, str]) -> set[str]:
    values = frame.loc[:, list(columns)].to_numpy()
    result: set[str] = set()
    for index, branch in enumerate(frame.index):
        dominated = any(
            np.all(values[other] >= values[index]) and np.any(values[other] > values[index])
            for other in range(len(values))
            if other != index
        )
        if not dominated:
            result.add(str(branch))
    return result


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family = pd.read_csv(REPORT_DIR / "family_deltas.csv")
    score = pd.read_csv(REPORT_DIR / "score_deltas.csv")
    geometry = pd.read_csv(REPORT_DIR / "geometry.csv")
    interaction = pd.read_csv(REPORT_DIR / "interaction_residuals.csv")
    return family, score, geometry, interaction


def build_summary(family: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    summary = family.groupby(["branch", "family"])["delta"].agg(["mean", "std"]).reset_index()
    mean_wide = summary.pivot(index="branch", columns="family", values="mean")
    std_wide = summary.pivot(index="branch", columns="family", values="std")
    ag = (
        score.loc[score.task == "AGNewsClassification"]
        .groupby("branch")["delta_vs_m0"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "ag_news_mean", "std": "ag_news_std"})
    )
    rows = []
    for branch in mean_wide.index:
        weights = weights_from_branch(branch)
        row: dict[str, object] = {"branch": branch, **{f"weight_{key}": value for key, value in weights.items()}}
        for family_name in mean_wide.columns:
            row[f"{family_name}_mean"] = mean_wide.loc[branch, family_name]
            row[f"{family_name}_std"] = std_wide.loc[branch, family_name]
        row.update(ag.loc[branch].to_dict())
        broad = [float(mean_wide.loc[branch, name]) for name in BROAD_FAMILIES]
        row["broad_mean"] = float(np.mean(broad))
        row["broad_worst"] = float(np.min(broad))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "mixture_summary.csv", index=False)
    return frame


def build_pairwise_pareto(summary: pd.DataFrame) -> pd.DataFrame:
    simplex = summary.loc[summary.branch != "control"].set_index("branch").copy()
    target_columns = {
        "sts": "sts_mean",
        "classification": "ag_news_mean",
        "retrieval": "retrieval_mean",
    }
    for left, right in combinations(OBJECTIVES, 2):
        front = nondominated(simplex, (target_columns[left], target_columns[right]))
        simplex[f"pareto_{left}_{right}"] = simplex.index.isin(front)
    simplex.reset_index().to_csv(DATA_DIR / "pairwise_pareto.csv", index=False)
    return simplex.reset_index()


def ternary_coordinates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame.weight_retrieval.to_numpy() + 0.5 * frame.weight_classification.to_numpy()
    y = math.sqrt(3) / 2 * frame.weight_classification.to_numpy()
    return x, y


def simplex_axes(ax: plt.Axes) -> None:
    triangle = np.asarray([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="#374151", linewidth=0.9)
    for value in (0.2, 0.4, 0.6, 0.8):
        ax.plot([value, 0.5 + 0.5 * value], [0, math.sqrt(3) / 2 * (1 - value)], color="#D1D5DB", lw=0.45)
        ax.plot([1 - value, 0.5 * (1 - value)], [0, math.sqrt(3) / 2 * (1 - value)], color="#D1D5DB", lw=0.45)
        ax.plot([0.5 * value, 1 - 0.5 * value], [math.sqrt(3) / 2 * value] * 2, color="#D1D5DB", lw=0.45)
    ax.text(-0.04, -0.035, "STS", ha="right", va="top", weight="bold")
    ax.text(1.04, -0.035, "Поиск", ha="left", va="top", weight="bold")
    ax.text(0.5, math.sqrt(3) / 2 + 0.045, "Классификация", ha="center", va="bottom", weight="bold")
    ax.set_xlim(-0.09, 1.09)
    ax.set_ylim(-0.08, 0.96)
    ax.set_aspect("equal")
    ax.axis("off")


def build_simplex_family_maps(summary: pd.DataFrame) -> None:
    simplex = summary.loc[summary.branch != "control"]
    x, y = ternary_coordinates(simplex)
    panels = [
        ("sts_mean", "STS"),
        ("classification_transfer_mean", "Перенос классификации"),
        ("clustering_mean", "Кластеризация"),
        ("retrieval_mean", "Поиск"),
    ]
    values = np.concatenate([simplex[column].to_numpy() for column, _ in panels])
    limit = float(np.max(np.abs(values)))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.4))
    scatter = None
    for ax, (column, title) in zip(axes.flat, panels, strict=True):
        simplex_axes(ax)
        scatter = ax.scatter(x, y, c=simplex[column], cmap="RdBu", vmin=-limit, vmax=limit, s=92, edgecolor="white", lw=0.7)
        ax.set_title(f"Δ {title}")
    assert scatter is not None
    cbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.035)
    cbar.set_label("Средняя дельта относительно M0")
    fig.suptitle("Перенос качества по симплексу обучающих целей", fontsize=12)
    fig.savefig(FIGURE_DIR / "simplex_family_maps.png")
    plt.close(fig)


def build_family_heatmap(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values(["weight_classification", "weight_sts", "weight_retrieval"], na_position="first")
    columns = [f"{family}_mean" for family in BROAD_FAMILIES]
    matrix = ordered[columns].to_numpy()
    limit = float(np.max(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(8.4, 10.3))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(columns)), [FAMILY_LABELS[name] for name in BROAD_FAMILIES], rotation=20, ha="right")
    ax.set_yticks(range(len(ordered)), ordered.branch)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=7.2, color="white" if abs(value) > limit * 0.5 else "#111827")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Δ относительно M0")
    ax.set_title("Межзадачный перенос при смешивании целей")
    fig.savefig(FIGURE_DIR / "family_delta_heatmap.png")
    plt.close(fig)


def build_pareto_plot(pareto: pd.DataFrame) -> None:
    targets = {
        "sts": ("sts_mean", "Δ STS"),
        "classification": ("ag_news_mean", "Δ AG News"),
        "retrieval": ("retrieval_mean", "Δ поиска"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, (left, right) in zip(axes, combinations(OBJECTIVES, 2), strict=True):
        x_column, x_label = targets[left]
        y_column, y_label = targets[right]
        flag = f"pareto_{left}_{right}"
        colors = np.where(pareto[flag], "#D1495B", "#AAB2BD")
        sizes = np.where(pareto[flag], 62, 30)
        ax.scatter(pareto[x_column], pareto[y_column], c=colors, s=sizes, alpha=0.9, edgecolor="white", lw=0.5)
        for _, row in pareto.loc[pareto[flag]].iterrows():
            label = row.branch.removeprefix("mix_").replace("_", "/")
            ax.annotate(label, (row[x_column], row[y_column]), xytext=(3, 3), textcoords="offset points", fontsize=6.3)
        ax.axhline(0, color="#6B7280", lw=0.7)
        ax.axvline(0, color="#6B7280", lw=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.18)
    fig.suptitle("Попарные Pareto-фронты целевых показателей", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pairwise_pareto.png")
    plt.close(fig)


def build_interaction_heatmap(interaction: pd.DataFrame) -> pd.DataFrame:
    summary = interaction.groupby(["branch", "family"])["interaction_residual"].agg(["mean", "std"]).reset_index()
    summary.to_csv(DATA_DIR / "interaction_residual_summary.csv", index=False)
    matrix = summary.pivot(index="branch", columns="family", values="mean").loc[:, list(BROAD_FAMILIES)]
    matrix = matrix.sort_values("retrieval", ascending=False)
    limit = float(np.max(np.abs(matrix.to_numpy())))
    fig, ax = plt.subplots(figsize=(8.5, 9.4))
    image = ax.imshow(matrix, cmap="PiYG", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [FAMILY_LABELS[name] for name in matrix.columns], rotation=20, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=7.0, color="white" if abs(value) > limit * 0.52 else "#111827")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Наблюдаемая Δ минус линейная интерполяция")
    ax.set_title("Нелинейный остаток смешивания")
    fig.savefig(FIGURE_DIR / "interaction_residual_heatmap.png")
    plt.close(fig)
    return summary


def build_geometry_maps(geometry: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    mean_geometry = geometry.groupby("branch").mean(numeric_only=True)
    baseline = mean_geometry.loc["m0"]
    delta = mean_geometry.drop(index="m0").subtract(baseline, axis="columns").reset_index()
    delta = delta.merge(summary[["branch", "weight_sts", "weight_classification", "weight_retrieval"]], on="branch", how="left")
    delta.to_csv(DATA_DIR / "geometry_delta_summary.csv", index=False)
    simplex = delta.loc[delta.branch != "control"]
    x, y = ternary_coordinates(simplex)
    panels = [
        ("effective_rank", "Эффективный ранг"),
        ("top10_eigenvalue_share", "Доля первых 10 компонент"),
        ("neighborhood_preservation", "Сохранение kNN-соседств"),
        ("retrieval_margin", "Query–document margin"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.4))
    for ax, (column, title) in zip(axes.flat, panels, strict=True):
        simplex_axes(ax)
        values = simplex[column].to_numpy()
        limit = float(np.max(np.abs(values)))
        scatter = ax.scatter(x, y, c=values, cmap="RdBu", vmin=-limit, vmax=limit, s=92, edgecolor="white", lw=0.7)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Δ")
        ax.set_title(title)
    fig.suptitle("Геометрические изменения по симплексу целей", fontsize=12)
    fig.savefig(FIGURE_DIR / "simplex_geometry_maps.png")
    plt.close(fig)
    return delta


def write_analysis_manifest(summary: pd.DataFrame, pareto: pd.DataFrame) -> None:
    best_maximin = summary.loc[summary.broad_worst.idxmax()]
    best_mean = summary.loc[summary.broad_mean.idxmax()]
    payload = {
        "input_directory": str(REPORT_DIR.relative_to(ROOT)),
        "mixture_count_including_control": int(len(summary)),
        "simplex_point_count": int((summary.branch != "control").sum()),
        "seeds": [42, 43, 44],
        "best_maximin": {"branch": best_maximin.branch, "worst_family_delta": float(best_maximin.broad_worst)},
        "best_broad_mean": {"branch": best_mean.branch, "mean_family_delta": float(best_mean.broad_mean)},
        "pairwise_front_sizes": {
            column.removeprefix("pareto_"): int(pareto[column].sum())
            for column in pareto.columns
            if column.startswith("pareto_")
        },
        "figures": sorted(path.name for path in FIGURE_DIR.glob("*.png")),
        "derived_tables": sorted(path.name for path in DATA_DIR.glob("*.csv")),
    }
    (ASSET_DIR / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    family, score, geometry, interaction = load_tables()
    summary = build_summary(family, score)
    pareto = build_pairwise_pareto(summary)
    build_simplex_family_maps(summary)
    build_family_heatmap(summary)
    build_pareto_plot(pareto)
    build_interaction_heatmap(interaction)
    build_geometry_maps(geometry, summary)
    write_analysis_manifest(summary, pareto)


if __name__ == "__main__":
    main()

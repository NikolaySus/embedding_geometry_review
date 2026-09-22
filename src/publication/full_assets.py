from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "english_geometry_pilot_v1" / "full"
ASSETS = ROOT / "build" / "full"
DATA = ASSETS / "data"
FIGURES = ASSETS / "figures"

FAMILIES = {
    "STS": {
        "BIOSSES",
        "STSBenchmark.clean_exact",
        "SICK-R.clean_exact",
        "STS12.clean_exact",
        "STS13.clean_exact",
        "STS14.clean_exact",
    },
    "Классификация": {
        "Banking77Classification",
        "EmotionClassification",
        "AmazonCounterfactualClassification",
        "MassiveIntentClassification",
        "AGNewsClassification",
    },
    "Кластеризация": {"TwentyNewsgroupsClustering.v2", "RedditClustering.v2"},
    "Поиск": {"SciFact", "NFCorpus", "ArguAna", "FiQA2018"},
}
MODEL_LABELS = {"all_minilm": "All-MiniLM"}
BRANCH_LABELS = {
    "control": "Контроль",
    "sts": "STS",
    "classification": "Классификация",
    "retrieval": "Поиск",
}
BRANCH_COLORS = {
    "control": "#6B7280",
    "sts": "#2374AB",
    "classification": "#C14953",
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


def family_delta_rows(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, seed, branch), group in deltas.groupby(["model", "seed", "branch"]):
        if branch == "m0" or model not in MODEL_LABELS:
            continue
        for family, tasks in FAMILIES.items():
            values = group.loc[group["task"].isin(tasks), "delta_vs_m0"]
            if len(values) != len(tasks):
                missing = tasks - set(group.loc[group["task"].isin(tasks), "task"])
                raise ValueError(f"Missing tasks for {model}/{seed}/{branch}/{family}: {sorted(missing)}")
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "branch": branch,
                    "family": family,
                    "delta": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def build_experiment_design() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x: float, y: float, width: float, height: float, title: str, body: str, color: str) -> None:
        ax.add_patch(
            plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="#263238", linewidth=1.0)
        )
        ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center", weight="bold")
        ax.text(x + width / 2, y + height * 0.30, body, ha="center", va="center", fontsize=8)

    box(0.2, 3.0, 2.5, 1.4, "All-MiniLM", "6 слоёв; готовая\nуниверсальная M0", "#E6E0F4")
    branch_y = [5.6, 4.1, 2.6, 1.1]
    branch_desc = {
        "control": "SimCSE",
        "sts": "CoSENT / STS",
        "classification": "SupCon / AG News",
        "retrieval": "MNRL / MS MARCO",
    }
    for branch, y in zip(BRANCH_LABELS, branch_y, strict=True):
        box(4.1, y, 2.6, 0.95, BRANCH_LABELS[branch], branch_desc[branch], "#F3F4F6")
        for source_y in (3.7,):
            ax.annotate(
                "",
                xy=(4.05, y + 0.48),
                xytext=(2.75, source_y),
                arrowprops={"arrowstyle": "->", "color": "#52606D", "lw": 0.9},
            )
    box(8.0, 4.45, 3.4, 1.55, "Прикладная оценка", "STS · классификация\nкластеризация · поиск", "#E3F0E8")
    box(8.0, 1.55, 3.4, 1.55, "Геометрическая диагностика", "спектральные агрегаты · kNN\nuniformity · margins", "#F5E8D7")
    for y in branch_y:
        ax.annotate("", xy=(7.95, 5.2), xytext=(6.75, y + 0.48), arrowprops={"arrowstyle": "->", "color": "#52606D", "lw": 0.8})
        ax.annotate("", xy=(7.95, 2.3), xytext=(6.75, y + 0.48), arrowprops={"arrowstyle": "->", "color": "#52606D", "lw": 0.8})
    ax.text(6.0, 0.25, "All-MiniLM × 5 состояний × 3 seed = 15 оценочных состояний", ha="center", weight="bold")
    fig.savefig(FIGURES / "experiment_design.png")
    plt.close(fig)


def build_transfer_heatmap(family_deltas: pd.DataFrame) -> pd.DataFrame:
    summary = family_deltas.groupby(["model", "branch", "family"])["delta"].agg(["mean", "std"]).reset_index()
    summary.to_csv(DATA / "family_transfer_summary.csv", index=False)
    row_order = [(model, branch) for model in MODEL_LABELS for branch in BRANCH_LABELS]
    matrix = np.asarray(
        [
            [
                summary.loc[
                    (summary.model == model) & (summary.branch == branch) & (summary.family == family),
                    "mean",
                ].iloc[0]
                for family in FAMILIES
            ]
            for model, branch in row_order
        ]
    )
    limit = float(np.max(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(FAMILIES)), list(FAMILIES))
    ax.set_yticks(
        range(len(row_order)),
        [f"{MODEL_LABELS[model]} · {BRANCH_LABELS[branch]}" for model, branch in row_order],
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if abs(value) > limit * 0.53 else "#111827"
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", color=color, fontsize=8)
    ax.axhline(3.5, color="white", linewidth=3)
    ax.set_title("Средняя дельта относительно собственной M0")
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Изменение основной метрики")
    fig.savefig(FIGURES / "family_transfer_heatmap.png")
    plt.close(fig)
    return summary


def build_target_transfer(family_deltas: pd.DataFrame) -> None:
    target_family = {"sts": "STS", "classification": "Классификация", "retrieval": "Поиск"}
    rows: list[dict[str, object]] = []
    for (model, seed, branch), group in family_deltas.groupby(["model", "seed", "branch"]):
        if branch not in target_family:
            continue
        target = target_family[branch]
        target_delta = float(group.loc[group.family == target, "delta"].iloc[0])
        other_delta = float(group.loc[group.family != target, "delta"].mean())
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "branch": branch,
                "target_family": target,
                "target_delta": target_delta,
                "non_target_mean_delta": other_delta,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA / "target_vs_non_target.csv", index=False)
    summary = frame.groupby(["model", "branch"], as_index=False).agg(
        target_delta=("target_delta", "mean"),
        target_std=("target_delta", "std"),
        non_target_delta=("non_target_mean_delta", "mean"),
        non_target_std=("non_target_mean_delta", "std"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for _, row in summary.iterrows():
        marker = "s"
        ax.errorbar(
            row.target_delta,
            row.non_target_delta,
            xerr=row.target_std,
            yerr=row.non_target_std,
            fmt=marker,
            markersize=8,
            capsize=3,
            color=BRANCH_COLORS[row.branch],
            label=f"{MODEL_LABELS[row.model]} · {BRANCH_LABELS[row.branch]}",
        )
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.axvline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("Изменение целевого семейства")
    ax.set_ylabel("Среднее изменение трёх нецелевых семейств")
    ax.set_title("Целевое улучшение и перенос на остальные задачи")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.savefig(FIGURES / "target_vs_non_target.png")
    plt.close(fig)


def load_trajectories() -> pd.DataFrame:
    frames = []
    for path in sorted(RUNS.glob("*/seed_*/*/geometry_trajectory.csv")):
        relative = path.relative_to(RUNS)
        model, seed_name, branch = relative.parts[:3]
        if branch == "m0":
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "branch", branch)
        frame.insert(0, "seed", int(seed_name.removeprefix("seed_")))
        frame.insert(0, "model", model)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(DATA / "geometry_trajectories.csv", index=False)
    return combined


def build_geometry_trajectories(trajectories: pd.DataFrame) -> None:
    metrics = [
        ("effective_rank", "Эффективный ранг"),
        ("top10_eigenvalue_share", "Доля первых 10 компонент"),
        ("mean_cosine", "Средний cosine"),
        ("neighborhood_preservation", "Сохранение kNN-соседств"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.8), sharex=True, squeeze=False)
    for row_index, model in enumerate(MODEL_LABELS):
        for col_index, (metric, title) in enumerate(metrics):
            ax = axes[row_index, col_index]
            for branch in BRANCH_LABELS:
                subset = trajectories[(trajectories.model == model) & (trajectories.branch == branch)]
                stats = subset.groupby("fraction")[metric].agg(["mean", "std"]).reset_index()
                ax.plot(stats.fraction, stats["mean"], marker="o", color=BRANCH_COLORS[branch], label=BRANCH_LABELS[branch])
                ax.fill_between(stats.fraction, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=BRANCH_COLORS[branch], alpha=0.13)
            ax.set_title(title)
            ax.grid(alpha=0.2)
            if col_index == 0:
                ax.set_ylabel(MODEL_LABELS[model])
            if row_index == 0:
                ax.set_xlabel("Доля шагов обучения")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    fig.suptitle("Траектории геометрии при одинаковом бюджете специализации", y=1.01, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURES / "geometry_trajectories.png")
    plt.close(fig)


def geometry_delta_frame(family_deltas: pd.DataFrame) -> pd.DataFrame:
    geometry = pd.read_csv(REPORTS / "pilot_v1_geometry.csv")
    geometry = geometry[geometry.model == "all_minilm"]
    baseline = geometry[geometry.branch == "m0"].set_index(["model", "seed"])
    records = []
    metric_columns = [column for column in geometry.columns if column not in {"model", "seed", "branch"}]
    for _, row in geometry[geometry.branch != "m0"].iterrows():
        base = baseline.loc[(row.model, row.seed)]
        record = {"model": row.model, "seed": int(row.seed), "branch": row.branch}
        for metric in metric_columns:
            record[f"delta_{metric}"] = float(row[metric] - base[metric])
        records.append(record)
    geo_delta = pd.DataFrame(records)
    family_wide = family_deltas.pivot(index=["model", "seed", "branch"], columns="family", values="delta").reset_index()
    merged = geo_delta.merge(family_wide, on=["model", "seed", "branch"], validate="one_to_one")
    merged.to_csv(DATA / "geometry_and_quality_deltas.csv", index=False)
    return merged


def build_geometry_correlations(merged: pd.DataFrame) -> None:
    panels = [
        ("delta_top_eigenvalue_share", "Классификация", "Δ доли первой компоненты", "Δ классификации", -0.996),
        ("delta_retrieval_margin", "Поиск", "Δ retrieval margin", "Δ поиска", 0.823),
        ("delta_effective_rank", "STS", "Δ effective rank", "Δ STS", 0.942),
        ("delta_effective_rank", "Поиск", "Δ effective rank", "Δ поиска", 0.981),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 8.0))
    for ax, (x_col, y_col, x_label, y_label, expected_r) in zip(axes.flat, panels, strict=True):
        for model, marker in (("all_minilm", "s"),):
            subset = merged[merged.model == model]
            for branch in BRANCH_LABELS:
                branch_rows = subset[subset.branch == branch]
                ax.scatter(branch_rows[x_col], branch_rows[y_col], marker=marker, s=43, color=BRANCH_COLORS[branch], alpha=0.85)
        coefficients = np.polyfit(merged[x_col], merged[y_col], 1)
        x_values = np.linspace(merged[x_col].min(), merged[x_col].max(), 100)
        ax.plot(x_values, np.polyval(coefficients, x_values), color="#374151", linewidth=1.0, linestyle="--")
        actual_r = float(np.corrcoef(merged[x_col], merged[y_col])[0, 1])
        if not np.isclose(actual_r, expected_r, atol=0.002):
            raise ValueError(f"Unexpected correlation for {x_col}/{y_col}: {actual_r}")
        ax.text(0.04, 0.94, f"r = {actual_r:+.3f}; n = 12", transform=ax.transAxes, va="top")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.18)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BRANCH_COLORS[branch], label=BRANCH_LABELS[branch], markersize=7)
        for branch in BRANCH_LABELS
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], marker="s", color="#374151", linestyle="none", label="All-MiniLM"),
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=6, frameon=False)
    fig.suptitle("Описательные связи изменений геометрии и качества", y=1.01, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIGURES / "geometry_quality_correlations.png")
    plt.close(fig)


def build_selected_task_deltas(deltas: pd.DataFrame) -> None:
    tasks = [
        ("STSBenchmark.clean_exact", "Clean STSBenchmark"),
        ("AGNewsClassification", "AG News"),
        ("TwentyNewsgroupsClustering.v2", "20 Newsgroups"),
        ("SciFact", "SciFact"),
    ]
    selected = deltas[deltas.task.isin(task for task, _ in tasks) & (deltas.branch != "m0")]
    summary = selected.groupby(["model", "branch", "task"]).delta_vs_m0.agg(["mean", "std"]).reset_index()
    summary.to_csv(DATA / "selected_task_deltas.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True)
    positions = np.arange(4)
    labels = [f"{MODEL_LABELS[model]} · {BRANCH_LABELS[branch]}" for model in MODEL_LABELS for branch in BRANCH_LABELS]
    for ax, (task, title) in zip(axes.flat, tasks, strict=True):
        rows = []
        for model in MODEL_LABELS:
            for branch in BRANCH_LABELS:
                rows.append(summary[(summary.model == model) & (summary.branch == branch) & (summary.task == task)].iloc[0])
        means = [row["mean"] for row in rows]
        stds = [row["std"] for row in rows]
        colors = [BRANCH_COLORS[row.branch] for row in rows]
        ax.barh(positions, means, xerr=stds, color=colors, alpha=0.88, capsize=2)
        ax.axvline(0, color="#374151", linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(positions, labels if ax in (axes[0, 0], axes[1, 0]) else [""] * 4, fontsize=7)
        ax.invert_yaxis()
    fig.suptitle("Изменения отдельных задач относительно собственной M0", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_task_deltas.png")
    plt.close(fig)


def build_overlap_audit() -> None:
    audit = json.loads((REPORTS / "pilot_v1_evaluation_overlap_audit.json").read_text(encoding="utf-8"))
    overlap_rows = []
    for result in audit["results"]:
        count = int(result["exact_overlap_counts"].get("sts", 0))
        if count:
            overlap_rows.append({"task": result["task"], "unique_text_overlap": count})
    summary_path = RUNS.parent / "mteb" / "full" / "all_minilm" / "seed_42" / "m0" / "evaluation_summary.json"
    clean = json.loads(summary_path.read_text(encoding="utf-8"))["CleanSTS"]
    clean_rows = [
        {
            "task": task.removesuffix(".clean_exact"),
            "remaining_pairs": int(values["rows"]),
            "removed_pairs": int(values["removed_rows"]),
        }
        for task, values in clean.items()
    ]
    overlap_frame = pd.DataFrame(overlap_rows)
    clean_frame = pd.DataFrame(clean_rows)
    overlap_frame.to_csv(DATA / "exact_text_overlaps.csv", index=False)
    clean_frame.to_csv(DATA / "clean_sts_pair_counts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(overlap_frame.task, overlap_frame.unique_text_overlap, color="#C14953")
    axes[0].set_title("Совпавшие уникальные тексты")
    axes[0].set_ylabel("Количество")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(clean_frame.task, clean_frame.remaining_pairs, label="Оставлено", color="#2A9D6F")
    axes[1].bar(clean_frame.task, clean_frame.removed_pairs, bottom=clean_frame.remaining_pairs, label="Удалено", color="#E9C46A")
    axes[1].set_title("Формирование clean_exact")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "overlap_audit.png")
    plt.close(fig)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    deltas = pd.read_csv(REPORTS / "pilot_v1_score_deltas.csv")
    deltas = deltas[deltas.model == "all_minilm"]
    family_deltas = family_delta_rows(deltas)
    family_deltas.to_csv(DATA / "family_deltas_by_seed.csv", index=False)
    build_experiment_design()
    build_transfer_heatmap(family_deltas)
    build_target_transfer(family_deltas)
    trajectories = load_trajectories()
    build_geometry_trajectories(trajectories)
    merged = geometry_delta_frame(family_deltas)
    correlations = merged.corr(numeric_only=True)
    correlations.to_csv(DATA / "geometry_correlations.csv")
    build_geometry_correlations(merged)
    build_selected_task_deltas(deltas)
    build_overlap_audit()
    print(f"Data: {DATA}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()


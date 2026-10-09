"""Publication evidence and figures derived only from the pinned unified run."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gallery.bundles import load_bundle
from gallery.data import Q, G, absolute_config
from gallery.tables import absolute_table
from gallery.surfaces import compact_surface

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = json.loads((ROOT / "configs/publication.json").read_text())
SOURCE = ROOT / "reports" / SETTINGS["experiment"] / "full"
MODEL = SOURCE / SETTINGS["model"]
CONFIG = json.loads((ROOT / "configs/unified_v1.json").read_text())
REGIMES = CONFIG["evaluation"]["trajectory_regimes"]
LABELS = ["Control", "100:0:0", "0:100:0", "0:0:100", "1:1:1", "20:0:80"]
COLORS = ["#666666", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#AA8800"]
STYLES = ["--", "-", "-", "-", ":", "-."]
FAMILY_LABELS = dict(zip(Q, ["AG News", "STS", "Classification transfer", "Clustering", "Retrieval"]))
COMPACT_IDS = ("1C-09-absolute-unit-dense", "2D-compact-dense", "3E-compact-dense", "4A-compact-dense", "5A-compact-dense")
FULL_IDS = ("design", "audit", "absolute", "geometry_simplex", "quality_simplex", "quality_trajectories", "geometry_trajectories", "spectra", "correlations", "pareto", "residuals")
AUDIT_FILES = sorted((ROOT / "runs" / SETTINGS["experiment"] / "full" / SETTINGS["model"] / "m0/evaluation").glob("*.clean_exact.json"))


def audit_counts():
    rows = []
    for path in AUDIT_FILES:
        record = json.loads(path.read_text())
        rows.append({"dataset": record["task"].removesuffix(".clean_exact"),
                     "removed": record["details"]["removed_rows"], "retained": record["details"]["rows"]})
    if len(rows) != 5:
        raise ValueError("Expected five clean STS audit records")
    return pd.DataFrame(rows)


def number(value, signed=False):
    return format(value, "+.4f" if signed else ".4f").replace(".", ",").replace("-", "−")


def correlations(delta):
    rows = []
    pure = delta[delta.branch.isin(REGIMES[:4])]
    for scope, data in [("endpoints12", pure), ("without_classification9", pure[pure.branch != REGIMES[2]])]:
        for x, y in [("effective_rank", "sts"), ("retrieval_margin", "retrieval")]:
            rows.append(dict(scope=scope, x=x, y=y, n=len(data), r=data[x].corr(data[y])))
    return pd.DataFrame(rows)


def trajectory_figure(kind="compact", dense=True, quality=None, geometry=None):
    quality = pd.read_csv(MODEL / "quality_trajectories.csv") if quality is None else quality
    geometry = pd.read_csv(MODEL / "trajectories.csv") if geometry is None else geometry
    if kind == "compact":
        panels = [(quality[quality.family == f], "delta", "Gate: Δ " + FAMILY_LABELS[f]) for f in ["sts", "retrieval"]]
        panels += [(geometry, "effective_rank", "Effective rank"), (geometry, "neighborhood_preservation", "kNN preservation")]
    elif kind == "quality":
        panels = [(quality[quality.family == f], "delta", "Gate: Δ " + FAMILY_LABELS[f]) for f in Q[1:] + Q[:1]]
    else:
        panels = [(geometry, key, label) for key, label in zip(G, ["Effective rank", "Top-10 variance share", "kNN preservation", "Retrieval margin (in-sample)"])]
    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(6.2, (5.2 if dense else 5.7) if rows == 2 else 6.8), squeeze=False)
    for ax, (frame, key, title) in zip(axes.flat, panels):
        for branch, label, color, style in zip(REGIMES, LABELS, COLORS, STYLES):
            values = frame[frame.branch == branch].groupby("step")[key].agg(["mean", "std"])
            ax.plot(values.index, values["mean"], label=label, color=color, linestyle=style, marker=".", markersize=3)
            ax.fill_between(values.index, values["mean"] - values["std"].fillna(0), values["mean"] + values["std"].fillna(0), color=color, alpha=.16)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 68, 135, 270])
        ax.set_xlim(0, 275)
        ax.set_xlabel("Training steps")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.11, right=.98, top=.93, bottom=.16, hspace=.50, wspace=.34)
    return fig


def normalize_caption(fig):
    for text in fig.texts:
        text.set_text(text.get_text().replace("Этап 1:", "Единый протокол:").replace("Этап 2:", "Единый протокол:").replace("M0: исходная модель; SD повторной оценки, не независимого обучения.", "M0: общая фиксированная оценка; не независимые повторы."))
    return fig


def shared_figures(bundle, data):
    delta = bundle["deltas"]
    mean = delta[delta.branch != "control"].groupby("branch")[Q + G].mean()
    variant = next(v for v in absolute_config()["variants"] if v.get("layout") == "dense")
    return [normalize_caption(absolute_table(bundle["absolute"], variant, data_dir=data)),
            normalize_caption(compact_surface(mean[G], bundle["triangles"], "2D", dense=True)[0]),
            normalize_caption(compact_surface(mean[Q[1:]], bundle["triangles"], "3E", dense=True)[0]),
            trajectory_figure(),
            normalize_caption(compact_surface(bundle["residual"], bundle["triangles"], "5A", dense=True)[0])]


def extra_figures(bundle):
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.axis("off")
    for y, text in [(1, "M0: all-MiniLM-L6-v2 (384D)"), (.75, "21 simplex points + barycenter + SimCSE control"), (.50, "23 regimes × 3 seeds = 69 runs; 270 updates/run, batch 64"), (.25, "Geometry: steps 0, 68, 135, 270 (all regimes)"), (0, "Quality: full at 270; gate at 0, 68, 135, 270 (6 regimes)")]:
        ax.text(.5, .1 + y*.8, text, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    design = fig
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    audit_data = audit_counts()
    ax.barh(audit_data.dataset, audit_data.retained, label="Retained", color="#009E73")
    ax.barh(audit_data.dataset, audit_data.removed, left=audit_data.retained, label="Removed", color="#D55E00")
    ax.set_xlabel("Evaluated text pairs")
    ax.legend(frameon=False)
    fig.tight_layout()
    audit = fig
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2))
    spectra = pd.read_csv(SOURCE / "spectra.csv")
    for branch, label, color in zip(["m0"] + REGIMES, ["M0"] + LABELS, ["black"] + COLORS):
        values = spectra[(spectra.branch == branch) & (spectra.step == (0 if branch == "m0" else 270))].copy()
        values["share"] = values.eigenvalue / values.groupby("seed", dropna=False).eigenvalue.transform("sum")
        mean = values.groupby("component").share.mean()
        axes[0].semilogy(mean.index, mean, color=color, label=label)
        axes[1].plot(mean.index, mean.cumsum(), color=color)
    # Resolve meaningful variance while retaining the near-zero spectral tail.
    axes[0].set_yscale("symlog", linthresh=1e-5)
    for ax, title in zip(axes, ["Normalized eigenvalues", "Cumulative variance"]):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Component")
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=.28, left=.1, right=.98, wspace=.3)
    spectral = fig
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.1))
    delta = bundle["deltas"]
    for ax, x, y in zip(axes, ["effective_rank", "retrieval_margin"], ["sts", "retrieval"]):
        for branch, label, color in zip(REGIMES[:4], LABELS[:4], COLORS[:4]):
            subset = delta[delta.branch == branch]
            ax.scatter(subset[x], subset[y], label=label, color=color, s=20)
        ax.set_xlabel("Δ " + ("Effective rank" if x == "effective_rank" else "Retrieval margin"))
        ax.set_ylabel("Δ " + FAMILY_LABELS[y])
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=.27, left=.11, right=.98, wspace=.4)
    correlation = fig
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.8))
    pareto = pd.read_csv(MODEL / "pareto.csv")
    for ax, family in zip(axes, ["sts", "AG News", "classification_transfer"]):
        part = pareto[pareto.x_family == family]
        ax.scatter(part.x_delta, part.y_delta, color="#888888", s=12)
        front = part[part.on_mean_front].sort_values("x_delta")
        ax.plot(front.x_delta, front.y_delta, ".-", color="#0072B2")
        ax.scatter([0], [0], marker="x", color="black", label="M0")
        ax.set_xlabel("Δ " + FAMILY_LABELS[family], fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Δ Retrieval")
    fig.tight_layout()
    return [design, audit, trajectory_figure("quality"), trajectory_figure("geometry"), spectral, correlation, fig]


def evidence(bundle, data):
    delta = bundle["deltas"]
    stats = delta.groupby("branch")[Q + G].agg(["mean", "std"])
    corr = correlations(delta)
    corr.to_csv(data / "correlations.csv", index=False)
    audit_counts().to_csv(data / "audit_counts.csv", index=False)
    values = {}
    for i, row in corr.iterrows():
        values[f"R{i}"] = number(row.r)
    absolute_mean = bundle["absolute"].groupby("branch")[Q + G].mean()
    for branch, prefix in [("m0", "M0"), ("control", "CONTROL"), (REGIMES[2], "CLS"), (REGIMES[3], "RET")]:
        for metric in Q + G:
            values[f"ABS_{prefix}_{metric}"] = number(absolute_mean.loc[branch, metric])
    balance = pd.read_csv(MODEL / "balance.csv").set_index("branch")
    values["CONTROL_MEAN_DELTA"] = number(balance.loc["control", "mean_delta"], True)
    values["M40_WORST_DELTA"] = number(balance.loc["mix_s040_c000_r060", "worst_family_delta"], True)
    for branch, prefix in [("control", "CONTROL"), (REGIMES[1], "STS"), (REGIMES[2], "CLS"), (REGIMES[3], "RET"), (REGIMES[5], "M20"), ("mix_s040_c000_r060", "M40")]:
        for metric in Q + G:
            values[f"{prefix}_{metric}"] = number(stats.loc[branch, (metric, "mean")], True)
    rows = ["| Режим STS:Cls:Ret | STS | Классификация (перенос) | Кластеризация | Поиск | AG News |", "|---|---:|---:|---:|---:|---:|"]
    for branch, label in zip(REGIMES, LABELS):
        rows.append("| " + label + " | " + " | ".join(number(stats.loc[branch, (m, "mean")], True) + " ± " + number(stats.loc[branch, (m, "std")]) for m in Q[1:] + Q[:1]) + " |")
    values["DELTA_TABLE"] = "\n".join(rows)
    residual = pd.read_csv(MODEL / "residuals.csv")
    rs = residual[residual.family == "retrieval"].groupby("branch")[["observed_delta", "expected_linear_delta", "interaction_residual"]].agg(["mean", "std"])
    rows = ["| Смесь | Наблюдаемая Δ поиска | Линейное ожидание | Остаток |", "|---|---:|---:|---:|"]
    for branch, label in [("mix_s000_c060_r040", "0:60:40"), ("mix_s000_c080_r020", "0:80:20"), (REGIMES[4], "1:1:1")]:
        rows.append("| " + label + " | " + " | ".join(number(rs.loc[branch, (m, "mean")], True) + " ± " + number(rs.loc[branch, (m, "std")]) for m in ["observed_delta", "expected_linear_delta", "interaction_residual"]) + " |")
    values["RESIDUAL_TABLE"] = "\n".join(rows)
    for metric in ["observed_delta", "expected_linear_delta", "interaction_residual"]:
        values["CENTER_" + metric] = number(rs.loc[REGIMES[4], (metric, "mean")], True)
    scores = pd.read_csv(SOURCE / "scores.csv")
    scores = scores[(scores.model == SETTINGS["model"]) & scores.branch.isin(["m0"] + REGIMES) & ((scores.step == 270) | (scores.branch == "m0"))]
    scores.to_csv(data / "selected_task_scores.csv", index=False)
    scorestats = scores.groupby(["task", "branch"]).score.agg(["mean", "std"])
    rows = ["| Задача | M0 | Control | STS | Cls | Ret | 1:1:1 | 20:0:80 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"AGNewsClassification": "AG News", "AmazonCounterfactualClassification": "Counterfactual", "Banking77Classification": "Banking77", "EmotionClassification": "Emotion", "MassiveIntentClassification": "MassiveIntent", "MSMARCORerankingGate": "MS MARCO gate", "TwentyNewsgroupsClustering.v2": "20 Newsgroups", "RedditClustering.v2": "Reddit", "STSBenchmark": "STS-B"}
    for task in sorted(scores.task.unique()):
        cells = []
        for branch in ["m0"] + REGIMES:
            stat = scorestats.loc[(task, branch)]
            cells.append(number(stat["mean"]) + ("" if branch == "m0" else " ± " + number(stat["std"])))
        base_task = task.removesuffix(".clean_exact")
        label = names.get(base_task, base_task) + (" clean" if task.endswith(".clean_exact") else "")
        rows.append("| " + label + " | " + " | ".join(cells) + " |")
    values["TASK_TABLE"] = "\n".join(rows)
    geometry = pd.read_csv(MODEL / "trajectories.csv")
    metrics = ["mean_cosine", "uniformity", "hubness_skew", "sts_alignment", "classification_intra_distance", "classification_inter_distance"]
    endpoint = geometry[(geometry.step == 270) & geometry.branch.isin(REGIMES)]
    gstats = endpoint.groupby("branch")[metrics].agg(["mean", "std"])
    gstats.to_csv(data / "additional_geometry_summary.csv")
    rows = ["| Режим | Косинус | Uniformity | Хабность | STS alignment | Внутриклассовое расстояние | Межклассовое расстояние |", "|---|---:|---:|---:|---:|---:|---:|"]
    for branch, label in zip(REGIMES, LABELS):
        rows.append("| " + label + " | " + " | ".join(number(gstats.loc[branch, (m, "mean")]) for m in metrics) + " |")
    values["GEOMETRY_TABLE"] = "\n".join(rows)
    (data / "evidence.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
    return values


def render_text(source, values):
    def substitute(match):
        key = match[1]
        return match[0] if key == "REFERENCES" else values[key]
    return re.sub(r"\{\{([^{}]+)\}\}", substitute, source)


def main(variant, output=None):
    output = output or ROOT / "build" / variant
    data, figures = output / "data", output / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(SETTINGS["experiment"], SETTINGS["model"])
    runs = pd.read_csv(SOURCE / "runs.csv")
    if set(runs.protocol) != {SETTINGS["protocol_hash"]}:
        raise ValueError("Publication protocol mismatch")
    files = list(MODEL.glob("*.csv")) + [MODEL / "bundle.json", SOURCE / "spectra.csv", SOURCE / "scores.csv", SOURCE / "runs.csv", ROOT / "configs/unified_v1.json"] + AUDIT_FILES
    hashes = {}
    for path in files:
        shutil.copyfile(path, data / path.name)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [ROOT / "configs/publication.json", Path(__file__), ROOT / "src/build_article.py"]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    values = evidence(bundle, data)
    (data / "provenance.json").write_text(json.dumps({**SETTINGS, "files": hashes}, indent=2) + "\n")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8}):
        shared = shared_figures(bundle, data)
        if variant == "compact":
            ids, plots = COMPACT_IDS, shared
        else:
            design, audit, quality, geometry, spectra, corr, pareto = extra_figures(bundle)
            plt.close(shared[3])
            ids, plots = FULL_IDS, [design, audit, shared[0], shared[1], shared[2], quality, geometry, spectra, corr, pareto, shared[4]]
        manifest = []
        for name, fig in zip(ids, plots):
            width, height = fig.get_size_inches()
            fig.savefig(figures / f"{name}.png", dpi=180)
            fig.savefig(figures / f"{name}.pdf")
            plt.close(fig)
            manifest.append(dict(id=name, width_inches=float(width), height_inches=float(height),
                                 provenance="provenance.json", experiment=SETTINGS["experiment"]))
        (data / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    return values

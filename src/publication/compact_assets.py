"""Build the five selected gallery figures directly for the compact article."""

from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gallery.data import G, Q, absolute_config, load_absolute_data, load_data
from gallery.surfaces import compact_surface
from gallery.tables import absolute_table
from gallery.trajectories import compact_trajectories

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/compact"
SOURCE = ROOT / "assets/compact/data"
MIX = ROOT / "reports/pilot_v2_mixtures"
FIGURE_IDS = (
    "1C-09-absolute-unit-dense", "2D-compact-dense", "3E-compact-dense",
    "4A-compact-dense", "5A-compact-dense",
)
EXAMPLE_BRANCHES = ("mix_s000_c060_r040", "mix_s000_c080_r020", "mix_s033_c033_r033")


def branch_label(branch):
    if branch == "mix_s033_c033_r033":
        return "1:1:1"
    return ":".join(str(int(part[1:])) for part in branch.removeprefix("mix_").split("_"))


def interaction_summary():
    residual = pd.read_csv(MIX / "interaction_residuals.csv")
    selected = residual[(residual.family == "retrieval") & residual.branch.isin(EXAMPLE_BRANCHES)]
    return selected.groupby("branch")[[
        "observed_delta", "expected_linear_delta", "interaction_residual",
    ]].agg(["mean", "std"]).loc[list(EXAMPLE_BRANCHES)]


def interaction_markdown_rows():
    rows = []
    for branch, values in interaction_summary().iterrows():
        cells = [branch_label(branch)]
        for metric in ("observed_delta", "expected_linear_delta", "interaction_residual"):
            cells.append(
                f"{values[(metric, 'mean')]:+.4f} ± {values[(metric, 'std')]:.4f}"
                .replace(".", ",").replace("-", "−")
            )
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def evidence():
    geometry = pd.read_csv(SOURCE / "geometry_quality.csv")
    correlations = [
        {"x": x, "y": y, "r": float(geometry[x].corr(geometry[y])), "n": len(geometry)}
        for x, y in [("delta_effective_rank", "STS"), ("delta_retrieval_margin", "Поиск")]
    ]
    families = pd.read_csv(MIX / "family_deltas.csv")
    scores = pd.read_csv(MIX / "score_deltas.csv")
    deltas = families.pivot(index=["branch", "seed"], columns="family", values="delta")
    deltas["ag"] = scores[scores.task == "AGNewsClassification"].set_index(["branch", "seed"]).delta_vs_m0
    stats = deltas.drop(index="control", level="branch").groupby("branch").agg(["mean", "std"])
    fronts = {}
    for axis in ("sts", "ag"):
        values = stats[[(axis, "mean"), ("retrieval", "mean")]].to_numpy()
        fronts[axis] = [
            branch_label(branch) for index, branch in enumerate(stats.index)
            if not np.any(np.all(values >= values[index], axis=1) & np.any(values > values[index], axis=1))
        ]
    return {"correlations": correlations, "pareto_fronts": fronts}, stats


def main():
    data, figures = OUT / "data", OUT / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    # Preserve earlier numerical detail even when its old visualization is retired.
    for path in SOURCE.iterdir():
        if path.suffix in {".csv", ".json"}:
            shutil.copyfile(path, data / path.name)
    details, stats = evidence()
    (data / "evidence.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    stats.to_csv(data / "pareto_summary.csv")
    interaction_summary().to_csv(data / "interaction_examples.csv")
    pd.DataFrame(details["correlations"]).to_csv(data / "correlations.csv", index=False)
    absolute = load_absolute_data()
    absolute.to_csv(data / "absolute_quality_geometry_by_seed.csv", index=False)
    absolute.groupby("branch")[Q + G].agg(["mean", "std"]).to_csv(data / "absolute_summary.csv")
    _, mean, residual, triangles = load_data()
    variant = next(v for v in absolute_config()["variants"] if v["id"] == FIGURE_IDS[0])
    manifest = []

    def save(fig, identity):
        width, height = fig.get_size_inches()
        fig.savefig(figures / f"{identity}.png", dpi=180)
        fig.savefig(figures / f"{identity}.pdf")
        plt.close(fig)
        manifest.append({"id": identity, "width_inches": float(width), "height_inches": float(height)})

    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8})
        save(absolute_table(absolute, variant, data_dir=data), FIGURE_IDS[0])
        save(compact_surface(mean[G], triangles, "2D", dense=True)[0], FIGURE_IDS[1])
        save(compact_surface(mean[Q[1:]], triangles, "3E", dense=True)[0], FIGURE_IDS[2])
        save(compact_trajectories(dense=True, data_dir=data), FIGURE_IDS[3])
        save(compact_surface(residual, triangles, "5A", dense=True)[0], FIGURE_IDS[4])
    (data / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for old in ("design", "quality_geometry", "diagnostics", "tradeoffs"):
        for suffix in ("png", "pdf"):
            (figures / f"{old}.{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    main()

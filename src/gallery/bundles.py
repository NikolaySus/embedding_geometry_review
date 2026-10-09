"""Explicit historical/unified adapters, independent of experiment execution."""

from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.backends.backend_pdf import PdfPages

from .data import ROOT, SRC, Q, G, absolute_config, load_absolute_data, load_data
from .tables import absolute_table
from .surfaces import compact_surface
from .trajectories import compact_trajectories


def load_bundle(experiment="historical", model="all_minilm"):
    if experiment == "historical":
        frame, _, residual, triangles = load_data()
        return {"absolute": load_absolute_data(), "deltas": frame,
                "residual": residual, "triangles": triangles,
                "trajectories": pd.read_csv(SRC / "geometry_trajectories.csv"), "historical": True}
    directory = ROOT / "reports" / experiment / "full" / model
    status = json.loads((directory.parent / "status.json").read_text())
    if not status["gallery_ready"]:
        raise ValueError("Incomplete experiment: gallery export is blocked")
    meta = json.loads((directory / "bundle.json").read_text())
    absolute = pd.read_csv(directory / "absolute.csv")
    delta = pd.read_csv(directory / "deltas.csv")
    mean = delta[delta.branch != "control"].groupby("branch")[Q+G].mean()
    weights = {s["name"]: s["weights"] for s in meta["regimes"]}
    coordinates = np.array([[weights[b]["retrieval"]+weights[b]["classification"]/2,
                             weights[b]["classification"]*np.sqrt(3)/2] for b in mean.index])
    residual = pd.read_csv(directory / "residuals.csv").groupby(["branch", "family"]).interaction_residual.mean().unstack().loc[mean.index, Q[1:]]
    trajectories = pd.read_csv(directory / "trajectories.csv")
    return {"absolute": absolute, "deltas": delta, "residual": residual,
            "triangles": Delaunay(coordinates).simplices, "trajectories": trajectories,
            "quality_trajectories": pd.read_csv(directory / "quality_trajectories.csv"),
            "historical": False, "metadata": meta}


def build_dense(experiment, model):
    bundle = load_bundle(experiment, model)
    output = ROOT / "build/gallery" / experiment / model
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    absolute, delta = bundle["absolute"], bundle["deltas"]
    mean = delta[delta.branch != "control"].groupby("branch")[Q+G].mean()
    scales = deepcopy(absolute_config())
    variant = next(v for v in scales["variants"] if v.get("layout") == "dense")
    if not bundle["historical"]:
        # Expand only when new observations exceed the old display bounds.
        for metric in G:
            lo, hi = scales["geometry"][metric]["range"]
            scales["geometry"][metric]["range"] = [min(lo, float(absolute[metric].min())), max(hi, float(absolute[metric].max()))]
        variant["quality_range"] = [min(0., float(absolute[Q].min().min())), max(1., float(absolute[Q].max().max()))]
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8})
        if bundle["historical"]:
            trajectory = compact_trajectories(True, data_dir=data, frame=bundle["trajectories"])
        else:
            from publication.unified_assets import trajectory_figure
            trajectory = trajectory_figure(quality=bundle["quality_trajectories"], geometry=bundle["trajectories"])
        figures = [
            ("1C-09-absolute-unit-dense", absolute_table(absolute, variant, data_dir=data, scale_config=scales)),
            ("2D-compact-dense", compact_surface(mean[G], bundle["triangles"], "2D", dense=True)[0]),
            ("3E-compact-dense", compact_surface(mean[Q[1:]], bundle["triangles"], "3E", dense=True)[0]),
            ("4A-compact-dense", trajectory),
            ("5A-compact-dense", compact_surface(bundle["residual"], bundle["triangles"], "5A", dense=True)[0])]
        with PdfPages(output / "DENSE.pdf") as pdf:
            for name, fig in figures:
                if not bundle["historical"]:
                    for text in fig.texts:
                        text.set_text(text.get_text().replace("Этап 1:", "Единый протокол:")
                            .replace("Этап 2:", "Единый протокол:")
                            .replace("M0: исходная модель; SD повторной оценки, не независимого обучения.",
                                     "M0: общая фиксированная оценка; повторение строки не является повтором опыта."))
                fig.savefig(output / "figures" / f"{name}.png", dpi=180)
                fig.savefig(output / "figures" / f"{name}.pdf")
                pdf.savefig(fig)
                plt.close(fig)
    absolute.to_csv(data / "absolute.csv", index=False)
    delta.to_csv(data / "deltas.csv", index=False)
    (output / "source.json").write_text(json.dumps({"experiment": experiment, "model": model,
        "historical": bundle["historical"], "metadata": bundle.get("metadata")}, ensure_ascii=False, indent=2))
    return output

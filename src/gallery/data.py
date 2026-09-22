"""Gallery paths, metric definitions, and measured mixture coordinates."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "gallery"
SRC = ASSETS / "data"
OUT = ROOT / "build" / "gallery"
WIDTH = 6.2
Q = ["AG News", "sts", "classification_transfer", "clustering", "retrieval"]
G = [
    "effective_rank",
    "top10_eigenvalue_share",
    "neighborhood_preservation",
    "retrieval_margin",
]
NAMES = ["AG", "STS", "Cls-tr", "Clust", "Ret", "Rank", "Top10", "kNN", "Margin"]
COLORS = {
    "control": "#666666",
    "sts": "#2475B0",
    "classification": "#BC434C",
    "retrieval": "#238568",
}

LABEL = dict(zip(Q + G, NAMES))


def short(b):
    if b == "control":
        return "Control"
    if b == "mix_s033_c033_r033":
        return "1:1:1"
    return ":".join(str(int(p[1:])) for p in b.removeprefix("mix_").split("_"))


def weights(b):
    if b == "mix_s033_c033_r033":
        return np.ones(3) / 3
    return np.array([int(p[1:]) for p in b.removeprefix("mix_").split("_")]) / 100


def xy(branches):
    w = np.array([weights(b) for b in branches])
    return w[:, 2] + w[:, 1] / 2, w[:, 1] * np.sqrt(3) / 2


def ternary_grid():
    lines = []
    for axis in range(3):
        for t in np.arange(0, 1.001, 0.2):
            a = np.zeros(3)
            b = np.zeros(3)
            a[axis] = b[axis] = t
            a[(axis + 1) % 3] = 1 - t
            b[(axis + 2) % 3] = 1 - t
            lines.append((a, b))
    return lines


def project(w):
    return np.array([w[2] + w[1] / 2, w[1] * np.sqrt(3) / 2])


def load_data():
    frame = pd.read_csv(SRC / "quality_geometry_by_seed.csv")
    mean = frame[frame.branch != "control"].groupby("branch")[Q + G].mean()
    residual = (
        pd.read_csv(SRC / "interaction_residuals.csv")
        .groupby(["branch", "family"])
        .interaction_residual.mean()
        .unstack()[Q[1:]]
        .loc[mean.index]
    )
    x, y = xy(mean.index)
    triangles = Delaunay(np.column_stack([x, y])).simplices
    return frame, mean, residual, triangles


def layout_config():
    return json.loads((ASSETS / "layouts.json").read_text())

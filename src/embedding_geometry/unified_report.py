"""Canonical observation tables and strict family-level gallery exports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT
from .mixtures import mixture_specs
from .unified_state import atomic_json, result_root


FAMILIES = {
    "AG News": ["AGNewsClassification"],
    "sts": ["BIOSSES", "STSBenchmark.clean_exact", "SICK-R.clean_exact", "STS12.clean_exact", "STS13.clean_exact", "STS14.clean_exact"],
    "classification_transfer": ["Banking77Classification", "EmotionClassification", "AmazonCounterfactualClassification", "MassiveIntentClassification"],
    "clustering": ["TwentyNewsgroupsClustering.v2", "RedditClustering.v2"],
    "retrieval": ["SciFact", "NFCorpus", "ArguAna", "FiQA2018"],
}
GEOMETRY = ["effective_rank", "top10_eigenvalue_share", "neighborhood_preservation", "retrieval_margin"]


def family_values(scores, members):
    if scores.task.duplicated().any():
        raise ValueError("Duplicate task scores")
    indexed = scores.set_index("task").score
    missing = set(members)-set(indexed.index)
    if missing:
        raise ValueError(f"Missing family members: {sorted(missing)}")
    values = indexed.loc[members].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite family scores")
    return float(values.mean())


def report(config, smoke=False):
    root = result_root(config, smoke)
    output = ROOT / "reports" / config["experiment_id"] / ("smoke" if smoke else "full")
    if smoke:
        output = output / root.name
    output.mkdir(parents=True, exist_ok=True)
    score_rows, geom_rows, run_rows, train_rows, spectra = [], [], [], [], []
    stored = json.loads((root / "protocol.json").read_text())
    if stored["config"] != config:
        raise ValueError("Report configuration does not match recorded experiment")
    protocol = stored["identity"]
    for path in sorted(root.rglob("geometry.json")):
        payload = json.loads(path.read_text())
        identity = payload["identity"]
        if identity["protocol"] != protocol:
            raise ValueError("Mixed protocol identities")
        geom_rows.append({**identity, "geometry_version": payload["version"],
                          "probe_ids_hash": payload["probe_ids_hash"],
                          "task_probe_scope": payload["task_probe_scope"], **payload["metrics"]})
        for i, value in enumerate(payload["covariance_eigenvalues"]):
            spectra.append({**{k: identity[k] for k in ("model", "branch", "seed", "step")},
                            "component": i+1, "eigenvalue": value})
    for path in sorted(root.rglob("evaluation/*.json")):
        payload = json.loads(path.read_text())
        if "task" not in payload:
            continue
        if payload["identity"]["protocol"] != protocol:
            raise ValueError("Mixed score protocols")
        score_rows.append({**payload["identity"], "task": payload["task"], "metric": payload["metric"],
                           "score": payload["score"], "seconds": payload["seconds"],
                           "filter": payload["details"].get("filter", "original"), "technical_only": smoke})
    for path in sorted(root.glob("*/seed_*/*/manifest.json")):
        payload = json.loads(path.read_text())
        run_rows.append({k: v for k, v in payload.items() if not isinstance(v, dict)})
        for row in json.loads((path.parent / "training.json").read_text()):
            train_rows.append({"model": payload["model"], "branch": payload["branch"], "seed": payload["seed"], **row})
    frames = {"scores": pd.DataFrame(score_rows), "geometry": pd.DataFrame(geom_rows),
              "runs": pd.DataFrame(run_rows), "training": pd.DataFrame(train_rows), "spectra": pd.DataFrame(spectra)}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    specs = mixture_specs(config)
    expected = len(specs)*len(config["models"])*len(config["seeds"])
    status = {"experiment_id": config["experiment_id"], "protocol": protocol, "technical_only": smoke,
              "expected_runs": expected, "complete_training_runs": len(run_rows), "gallery_ready": False}
    if smoke:
        status["note"] = "Technical smoke only: no scientific family aggregates or interpolated surfaces"
    else:
        try:
            if len(run_rows) != expected:
                raise ValueError(f"Expected {expected} completed training manifests, found {len(run_rows)}")
            for model in config["models"]:
                export_gallery(config, frames, specs, model, output / model)
            status["gallery_ready"] = True
        except ValueError as exc:
            status["incomplete_reason"] = str(exc)
    atomic_json(output / "status.json", status)
    return status


def export_gallery(config, frames, specs, model, output):
    scores = frames["scores"]
    geom = frames["geometry"]
    if scores.empty or geom.empty:
        raise ValueError("No evaluation/geometry observations")
    scores, geom = scores[scores.model == model], geom[geom.model == model]
    keys = ["branch", "seed", "step"]
    if scores.duplicated(keys+["task"]).any() or geom.duplicated(keys).any():
        raise ValueError("Duplicate observation keys")
    if geom.geometry_version.nunique() != 1 or geom.probe_ids_hash.nunique() != 1:
        raise ValueError("Mixed geometry definitions or samples")
    total = config["training"]["total_steps"]
    base_scores = scores[scores.branch == "m0"]
    base_geom = geom[geom.branch == "m0"]
    if len(base_geom) != 1:
        raise ValueError("Missing/duplicate M0 geometry")
    base = {family: family_values(base_scores, tasks) for family, tasks in FAMILIES.items()}
    base.update(base_geom.iloc[0][GEOMETRY].to_dict())
    absolute, deltas, residuals, trajectories, quality_trajectories = [], [], [], [], []
    endpoints = {"sts": "mix_s100_c000_r000", "classification": "mix_s000_c100_r000", "retrieval": "mix_s000_c000_r100"}
    for seed in config["seeds"]:
        absolute.append({"branch": "m0", "seed": seed, **base})
        seed_values = {}
        for spec in specs:
            name = spec["name"]
            s = scores[(scores.branch == name) & (scores.seed == seed) & (scores.step == total)]
            g = geom[(geom.branch == name) & (geom.seed == seed) & (geom.step == total)]
            if len(g) != 1:
                raise ValueError(f"Incomplete final geometry: {model}/{seed}/{name}")
            values = {family: family_values(s, tasks) for family, tasks in FAMILIES.items()}
            values.update(g.iloc[0][GEOMETRY].to_dict())
            absolute.append({"branch": name, "seed": seed, **values})
            difference = {key: values[key]-base[key] for key in base}
            seed_values[name] = difference
            deltas.append({"branch": name, "seed": seed, **difference})
            for step in config["training"]["snapshot_steps"]:
                snap = geom[(geom.branch == name) & (geom.seed == seed) & (geom.step == step)]
                if len(snap) != 1:
                    raise ValueError(f"Missing trajectory observation: {name}/{seed}/{step}")
                trajectories.append({"branch": name, "seed": seed, "step": step,
                    **snap.iloc[0].drop(labels=[k for k in snap.columns if k in keys]).to_dict()})
                if name in config["evaluation"]["trajectory_regimes"]:
                    subset = base_scores if step == 0 else scores[(scores.branch == name) & (scores.seed == seed) & (scores.step == step)]
                    for family, tasks in FAMILIES.items():
                        gate = [task for task in tasks if task.removesuffix(".clean_exact") in config["evaluation"]["gate_tasks"] or task == "AGNewsClassification"]
                        value = family_values(subset, gate)
                        quality_trajectories.append({"branch": name, "seed": seed, "step": step, "family": family,
                            "suite": "gate", "score": value, "delta": value-family_values(base_scores, gate)})
        for spec in specs:
            if spec["name"] == "control":
                continue
            for family in list(FAMILIES)[1:]:
                observed = seed_values[spec["name"]][family]
                expected = sum(spec["weights"].get(key, 0)*seed_values[value][family] for key, value in endpoints.items())
                residuals.append({"branch": spec["name"], "seed": seed, "family": family,
                                  "observed_delta": observed, "expected_linear_delta": expected,
                                  "interaction_residual": observed-expected})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("absolute", absolute), ("deltas", deltas), ("residuals", residuals),
                       ("trajectories", trajectories), ("quality_trajectories", quality_trajectories)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(absolute).groupby("branch")[list(FAMILIES)+GEOMETRY].agg(["mean", "std"]).to_csv(output / "summary.csv")
    delta_frame = pd.DataFrame(deltas)
    means = delta_frame.groupby("branch")[list(FAMILIES)].mean()
    independent = list(FAMILIES)[1:]
    pd.DataFrame({"mean_delta": means[independent].mean(axis=1),
                  "worst_family_delta": means[independent].min(axis=1)}).to_csv(output / "balance.csv")
    points = means.drop(index="control")
    fronts = []
    for axis in ("sts", "AG News", "classification_transfer"):
        values = points[[axis, "retrieval"]].to_numpy()
        for i, branch in enumerate(points.index):
            nondominated = not np.any(np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1))
            fronts.append({"branch": branch, "x_family": axis, "y_family": "retrieval",
                           "x_delta": values[i, 0], "y_delta": values[i, 1],
                           "on_mean_front": nondominated})
    pd.DataFrame(fronts).to_csv(output / "pareto.csv", index=False)
    atomic_json(output / "bundle.json", {"schema_version": 1, "model": model, "experiment_id": config["experiment_id"],
        "seeds": config["seeds"], "geometry_version": geom.geometry_version.iloc[0],
        "baseline": "Shared frozen M0 at fixed evaluation seed; replicated only for paired deltas, not independent repeats",
        "regimes": specs, "complete": True})

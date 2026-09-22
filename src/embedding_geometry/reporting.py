from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT


def build_report(config: dict[str, Any]) -> Path:
    experiment_root = ROOT / "runs" / config["experiment_id"]
    full_summaries = list(experiment_root.glob("mteb/full/*/seed_*/*/evaluation_summary.json"))
    summaries = full_summaries or list(experiment_root.glob("mteb/gate/*/seed_*/*/evaluation_summary.json"))
    suite = "full" if full_summaries else "gate"
    records = []
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task, result in payload["mteb"].items():
            records.append({"suite": suite, "model": payload["model"], "seed": payload["seed"], "branch": payload["branch"], "task": task, "score": result["score"]})
        records.append({"suite": suite, "model": payload["model"], "seed": payload["seed"], "branch": payload["branch"], "task": "AGNewsClassification", "score": payload["AGNewsClassification"]["accuracy"]})
        records.append({"suite": suite, "model": payload["model"], "seed": payload["seed"], "branch": payload["branch"], "task": "MSMARCORerankingGate", "score": payload["MSMARCORerankingGate"]["ndcg_at_10"]})
        for task, result in payload.get("CleanSTS", {}).items():
            records.append({"suite": suite, "model": payload["model"], "seed": payload["seed"], "branch": payload["branch"], "task": task, "score": result["cosine_spearman"]})

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "pilot_v1_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["suite", "model", "seed", "branch", "task", "score"])
        writer.writeheader()
        writer.writerows(records)

    baseline = {
        (row["model"], row["seed"], row["task"]): row["score"]
        for row in records
        if row["branch"] == "m0"
    }
    deltas = [
        {**row, "delta_vs_m0": row["score"] - baseline[(row["model"], row["seed"], row["task"])]}
        for row in records
    ]
    with (report_dir / "pilot_v1_score_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*records[0], "delta_vs_m0"])
        writer.writeheader()
        writer.writerows(deltas)

    geometry_rows = []
    for model_name in config["models"]:
        for seed in config["seeds"]:
            run_root = experiment_root / "full" / model_name / f"seed_{seed}"
            m0_path = run_root / "m0_geometry.json"
            if not m0_path.exists():
                continue
            geometry_rows.append({"model": model_name, "seed": seed, "branch": "m0", **json.loads(m0_path.read_text())})
            for branch in ("control", "sts", "classification", "retrieval"):
                task_path = run_root / branch / "task_geometry.json"
                trajectory_path = run_root / branch / "geometry_trajectory.csv"
                if not task_path.exists() or not trajectory_path.exists():
                    continue
                with trajectory_path.open(encoding="utf-8") as handle:
                    trajectory = list(csv.DictReader(handle))
                final = {key: float(value) for key, value in trajectory[-1].items() if key not in {"step", "fraction"}}
                geometry_rows.append({"model": model_name, "seed": seed, "branch": branch, **final, **json.loads(task_path.read_text())})
    geometry_fields = ["model", "seed", "branch", *sorted(set().union(*(row.keys() for row in geometry_rows)) - {"model", "seed", "branch"})]
    with (report_dir / "pilot_v1_geometry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=geometry_fields)
        writer.writeheader()
        writer.writerows(geometry_rows)

    families = {
        "sts": {"BIOSSES", "STSBenchmark.clean_exact", "SICK-R.clean_exact", "STS12.clean_exact", "STS13.clean_exact", "STS14.clean_exact"},
        "classification": {"Banking77Classification", "EmotionClassification", "AmazonCounterfactualClassification", "MassiveIntentClassification", "AGNewsClassification"},
        "clustering": {"TwentyNewsgroupsClustering.v2", "RedditClustering.v2"},
        "retrieval": {"SciFact", "NFCorpus", "ArguAna", "FiQA2018"},
    }
    family_deltas = {}
    for model_name in config["models"]:
        for seed in config["seeds"]:
            for branch in ("control", "sts", "classification", "retrieval"):
                rows = [row for row in deltas if row["model"] == model_name and row["seed"] == seed and row["branch"] == branch]
                for family, tasks in families.items():
                    values = [row["delta_vs_m0"] for row in rows if row["task"] in tasks]
                    family_deltas[(model_name, seed, branch, family)] = float(np.mean(values))

    geometry_by_key = {(row["model"], row["seed"], row["branch"]): row for row in geometry_rows}
    correlations = []
    metric_names = [field for field in geometry_fields if field not in {"model", "seed", "branch"}]
    for family in families:
        for metric in metric_names:
            x, y = [], []
            for key, row in geometry_by_key.items():
                model_name, seed, branch = key
                if branch == "m0":
                    continue
                base = geometry_by_key[(model_name, seed, "m0")]
                if metric not in row or metric not in base:
                    continue
                x.append(float(row[metric]) - float(base[metric]))
                y.append(family_deltas[(model_name, seed, branch, family)])
            if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                correlations.append({"family": family, "geometry_metric": metric, "pearson_r": float(np.corrcoef(x, y)[0, 1]), "n": len(x)})
    correlations.sort(key=lambda row: abs(row["pearson_r"]), reverse=True)
    with (report_dir / "pilot_v1_geometry_correlations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "geometry_metric", "pearson_r", "n"])
        writer.writeheader()
        writer.writerows(correlations)

    grouped = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["branch"], row["task"])].append(row["score"])
    lines = [
        "# English Embedding Geometry Pilot v1",
        "",
        f"Scores use the `{suite}` suite and are reported across completed seeds. MTEB task scores use each task's declared main metric.",
        "",
        "| Model | Branch | Task | Mean | Std | Seeds |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, values in sorted(grouped.items()):
        model, branch, task = key
        lines.append(f"| {model} | {branch} | {task} | {np.mean(values):.6f} | {np.std(values):.6f} | {len(values)} |")
    lines.extend(["", "## Mean deltas versus M0 by task family", "", "| Model | Branch | STS | Classification | Clustering | Retrieval |", "|---|---|---:|---:|---:|---:|"])
    for model_name in config["models"]:
        for branch in ("control", "sts", "classification", "retrieval"):
            values = {family: np.mean([family_deltas[(model_name, seed, branch, family)] for seed in config["seeds"]]) for family in families}
            lines.append(f"| {model_name} | {branch} | {values['sts']:+.6f} | {values['classification']:+.6f} | {values['clustering']:+.6f} | {values['retrieval']:+.6f} |")
    lines.extend(["", "## Strongest geometry/performance associations", "", "Exploratory Pearson correlations across model, seed, and branch deltas; these are descriptive, not causal.", "", "| Family | Geometry metric | Pearson r | N |", "|---|---|---:|---:|"])
    for row in correlations[:20]:
        lines.append(f"| {row['family']} | {row['geometry_metric']} | {row['pearson_r']:+.4f} | {row['n']} |")
    report_path = report_dir / "pilot_v1.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

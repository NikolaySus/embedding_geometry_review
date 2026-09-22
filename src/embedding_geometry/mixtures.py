from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .common import ROOT, check_disk, device, read_jsonl, set_seed, write_json
from .evaluation import evaluate_checkpoint, evaluate_clean_sts
from .geometry import append_snapshot, generic_snapshot, task_geometry
from .modeling import load_encoder, verify_normalized
from .training import (
    _batches,
    _cosent_loss,
    _retrieval_loss,
    _save_model,
    _schedule,
    _simcse_loss,
    _supervised_contrastive_loss,
)


OBJECTIVES = ("sts", "classification", "retrieval")
FAMILIES = {
    "sts": {
        "BIOSSES",
        "STSBenchmark.clean_exact",
        "SICK-R.clean_exact",
        "STS12.clean_exact",
        "STS13.clean_exact",
        "STS14.clean_exact",
    },
    "classification": {
        "Banking77Classification",
        "EmotionClassification",
        "AmazonCounterfactualClassification",
        "MassiveIntentClassification",
        "AGNewsClassification",
    },
    "classification_transfer": {
        "Banking77Classification",
        "EmotionClassification",
        "AmazonCounterfactualClassification",
        "MassiveIntentClassification",
    },
    "clustering": {"TwentyNewsgroupsClustering.v2", "RedditClustering.v2"},
    "retrieval": {"SciFact", "NFCorpus", "ArguAna", "FiQA2018"},
}


def simplex_weights(denominator: int = 5) -> list[tuple[int, int, int]]:
    if denominator <= 0:
        raise ValueError("simplex denominator must be positive")
    return [
        (sts, classification, denominator - sts - classification)
        for sts in range(denominator + 1)
        for classification in range(denominator - sts + 1)
    ]


def mixture_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    denominator = int(config["mixtures"]["simplex_denominator"])
    specs = []
    if config["mixtures"].get("include_control", False):
        specs.append({"name": "control", "weights": {"control": 1.0}})
    for sts, classification, retrieval in simplex_weights(denominator):
        specs.append(
            {
                "name": f"mix_s{sts * 100 // denominator:03d}_c{classification * 100 // denominator:03d}_r{retrieval * 100 // denominator:03d}",
                "weights": {
                    "sts": sts / denominator,
                    "classification": classification / denominator,
                    "retrieval": retrieval / denominator,
                },
            }
        )
    if config["mixtures"].get("include_barycenter", False):
        specs.append(
            {
                "name": "mix_s033_c033_r033",
                "weights": {"sts": 1 / 3, "classification": 1 / 3, "retrieval": 1 / 3},
            }
        )
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("mixture names are not unique")
    return specs


def step_counts(weights: dict[str, float], total_steps: int) -> dict[str, int]:
    if set(weights) == {"control"}:
        return {"control": total_steps}
    if set(weights) != set(OBJECTIVES):
        raise ValueError(f"Unexpected objective set: {sorted(weights)}")
    raw = {name: weights[name] * total_steps for name in OBJECTIVES}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = total_steps - sum(counts.values())
    order = sorted(OBJECTIVES, key=lambda name: (-(raw[name] - counts[name]), OBJECTIVES.index(name)))
    for name in order[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != total_steps:
        raise AssertionError("step allocation does not match total_steps")
    return counts


def objective_schedule(weights: dict[str, float], total_steps: int, seed: int) -> list[str]:
    counts = step_counts(weights, total_steps)
    schedule = [objective for objective, count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(schedule)
    return schedule


def interpolation_residual(observed: float, weights: dict[str, float], pure_deltas: dict[str, float]) -> float:
    expected = sum(weights[objective] * pure_deltas[objective] for objective in OBJECTIVES)
    return observed - expected


def _batch_stream(rows: list[dict[str, Any]], batch_size: int, seed: int, objective: str) -> Iterator[list[dict[str, Any]]]:
    epoch = 0
    while True:
        yielded = False
        for batch in _batches(rows, batch_size, 1, seed + epoch * 10_007, objective):
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError(f"No complete batches available for {objective}")
        epoch += 1


def _loss(model, objective: str, batch, torch_device: torch.device, config: dict[str, Any]):
    if objective == "control":
        return _simcse_loss(model, batch, torch_device, config["training"]["temperature"])
    if objective == "sts":
        return _cosent_loss(model, batch, torch_device, config["training"]["cosent_scale"])
    if objective == "classification":
        return _supervised_contrastive_loss(model, batch, torch_device, config["training"]["temperature"])
    if objective == "retrieval":
        return _retrieval_loss(model, batch, torch_device, config["training"]["temperature"])
    raise ValueError(f"Unknown objective: {objective}")


def train_mixture(
    config: dict[str, Any],
    *,
    model_name: str,
    seed: int,
    spec: dict[str, Any],
    smoke: bool = False,
) -> Path:
    set_seed(seed)
    mode = "smoke" if smoke else "full"
    run_root = ROOT / "runs" / config["experiment_id"] / mode / model_name / f"seed_{seed}"
    output_dir = run_root / spec["name"]
    final_model = output_dir / "final_model"
    if final_model.exists() and (output_dir / "manifest.json").exists():
        return final_model

    model_cfg = config["models"][model_name]
    model = load_encoder(
        model_cfg["model_id"],
        max_seq_length=model_cfg["max_seq_length"],
        device=device(),
        raw_backbone=False,
    )
    torch_device = device()
    model.to(torch_device)
    model.train()

    total_steps = config["training"]["smoke_steps"] if smoke else config["training"]["total_steps"]
    schedule = objective_schedule(spec["weights"], total_steps, seed)
    rows = {
        "control": read_jsonl("data/processed/neutral_control.jsonl"),
        "sts": read_jsonl("data/processed/sts.jsonl"),
        "classification": read_jsonl("data/processed/classification.jsonl"),
        "retrieval": read_jsonl("data/processed/retrieval.jsonl"),
    }
    streams = {
        objective: _batch_stream(rows[objective], config["training"]["batch_size"], seed, objective)
        for objective in set(schedule)
    }
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = _schedule(optimizer, total_steps, config["training"]["warmup_ratio"])
    diagnostics_config = config
    fractions = config["training"]["trajectory_fractions"]
    if smoke:
        diagnostics_config = deepcopy(config)
        diagnostics_config["geometry"]["generic_sample_size"] = min(512, config["geometry"]["generic_sample_size"])
        diagnostics_config["geometry"]["pair_sample_size"] = min(5_000, config["geometry"]["pair_sample_size"])
        fractions = [0.0, 1.0]
    snapshot_steps = {round(fraction * total_steps): fraction for fraction in fractions}
    reference_path = run_root / "m0_reference_embeddings.npy"
    if not reference_path.exists():
        baseline = load_encoder(model_cfg["model_id"], max_seq_length=model_cfg["max_seq_length"], device=torch_device)
        baseline_geometry = generic_snapshot(
            baseline,
            diagnostics_config,
            seed=seed,
            save_reference_path=reference_path,
        )
        write_json(run_root / "m0_geometry.json", {**baseline_geometry, **task_geometry(baseline, diagnostics_config, seed)})
        del baseline

    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = output_dir / "geometry_trajectory.csv"
    metrics_path = output_dir / "train_metrics.jsonl"
    if geometry_path.exists():
        geometry_path.unlink()
    started = time.time()
    losses: list[float] = []

    def snapshot(step: int, fraction: float) -> None:
        model.eval()
        metrics = generic_snapshot(model, diagnostics_config, seed=seed, reference_path=reference_path)
        append_snapshot(geometry_path, {"step": step, "fraction": fraction, **metrics})
        model.train()

    if 0 in snapshot_steps:
        snapshot(0, snapshot_steps[0])

    with metrics_path.open("w", encoding="utf-8") as log:
        for step, objective in enumerate(tqdm(schedule, desc=f"{spec['name']} seed={seed}"), start=1):
            optimizer.zero_grad(set_to_none=True)
            dtype = torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float32
            with torch.autocast(device_type=torch_device.type, dtype=dtype, enabled=torch_device.type == "cuda"):
                loss = _loss(model, objective, next(streams[objective]), torch_device, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            value = float(loss.detach())
            losses.append(value)
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "objective": objective,
                        "loss": value,
                        "grad_norm": float(grad_norm),
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                + "\n"
            )
            log.flush()
            if step in snapshot_steps:
                snapshot(step, snapshot_steps[step])

    model.eval()
    write_json(output_dir / "task_geometry.json", task_geometry(model, diagnostics_config, seed))
    result = {
        "name": spec["name"],
        "weights": spec["weights"],
        "seed": seed,
        "steps": total_steps,
        "step_counts": dict(Counter(schedule)),
        "schedule_sha256": hashlib.sha256("\n".join(schedule).encode()).hexdigest(),
        "examples": total_steps * config["training"]["batch_size"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "elapsed_seconds": time.time() - started,
        "embedding_norms": verify_normalized(model, ["A short diagnostic sentence.", "A different diagnostic sentence."]),
        "technical_status": "pass",
    }
    write_json(output_dir / "manifest.json", result)
    _save_model(model, final_model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_model


def _model_digest(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(model_dir)).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _summary_path(config: dict[str, Any], model_name: str, seed: int, branch: str) -> Path:
    return ROOT / "runs" / config["experiment_id"] / "mteb" / "full" / model_name / f"seed_{seed}" / branch / "evaluation_summary.json"


def _complete_summary(path: Path, expected_tasks: int) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        len(payload.get("mteb", {})) == expected_tasks
        and len(payload.get("CleanSTS", {})) == 5
        and "AGNewsClassification" in payload
        and "MSMARCORerankingGate" in payload
    )


def evaluate_and_prune(config: dict[str, Any], model_name: str, seed: int, spec: dict[str, Any], model_path: str | Path) -> None:
    branch = spec["name"]
    summary_path = _summary_path(config, model_name, seed, branch)
    expected_tasks = len(config["evaluation"]["full_tasks"])
    if not _complete_summary(summary_path, expected_tasks):
        payload = evaluate_checkpoint(
            config,
            model_path,
            model_name=model_name,
            seed=seed,
            branch=branch,
            suite="full",
        )
        model = load_encoder(
            model_path,
            max_seq_length=config["models"][model_name]["max_seq_length"],
            device=device(),
        )
        payload["CleanSTS"] = evaluate_clean_sts(model, config)
        payload["mixture_weights"] = spec["weights"]
        write_json(summary_path, payload)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not _complete_summary(summary_path, expected_tasks):
        raise RuntimeError(f"Incomplete evaluation: {summary_path}")

    model_dir = Path(model_path)
    output_dir = model_dir.parent
    state = {
        "name": branch,
        "seed": seed,
        "weights": spec["weights"],
        "evaluation_complete": True,
        "summary": str(summary_path.relative_to(ROOT)),
        "model_sha256": _model_digest(model_dir),
        "model_retained": not config["mixtures"].get("prune_after_evaluation", True),
    }
    if config["mixtures"].get("prune_after_evaluation", True):
        shutil.rmtree(model_dir)
        state["model_retained"] = False
    write_json(output_dir / "run_state.json", state)


def evaluate_m0(config: dict[str, Any], model_name: str, seed: int) -> None:
    summary_path = _summary_path(config, model_name, seed, "m0")
    expected_tasks = len(config["evaluation"]["full_tasks"])
    if _complete_summary(summary_path, expected_tasks):
        return
    source = config["models"][model_name]["model_id"]
    payload = evaluate_checkpoint(config, source, model_name=model_name, seed=seed, branch="m0", suite="full")
    model = load_encoder(source, max_seq_length=config["models"][model_name]["max_seq_length"], device=device())
    payload["CleanSTS"] = evaluate_clean_sts(model, config)
    write_json(summary_path, payload)


def run_mixtures(config: dict[str, Any], *, smoke: bool = False) -> None:
    check_disk(config["storage"]["minimum_free_gb"])
    model_name = next(iter(config["models"]))
    specs = mixture_specs(config)
    if smoke:
        by_name = {spec["name"]: spec for spec in specs}
        selected = [by_name["mix_s020_c020_r060"], by_name["mix_s033_c033_r033"]]
        for spec in selected:
            path = train_mixture(config, model_name=model_name, seed=config["seeds"][0], spec=spec, smoke=True)
            manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
            if manifest["technical_status"] != "pass":
                raise RuntimeError(f"Smoke validation failed for {spec['name']}")
            shutil.rmtree(path)
        return

    for seed in config["seeds"]:
        evaluate_m0(config, model_name, seed)
        for spec in specs:
            output_dir = ROOT / "runs" / config["experiment_id"] / "full" / model_name / f"seed_{seed}" / spec["name"]
            state_path = output_dir / "run_state.json"
            summary_path = _summary_path(config, model_name, seed, spec["name"])
            if state_path.exists() and _complete_summary(summary_path, len(config["evaluation"]["full_tasks"])):
                continue
            final_model = output_dir / "final_model"
            if not final_model.exists():
                final_model = train_mixture(config, model_name=model_name, seed=seed, spec=spec)
            evaluate_and_prune(config, model_name, seed, spec, final_model)


def build_mixture_report(config: dict[str, Any]) -> Path:
    model_name = next(iter(config["models"]))
    specs = mixture_specs(config)
    records = []
    geometry_rows = []
    for seed in config["seeds"]:
        branches = ["m0", *(spec["name"] for spec in specs)]
        for branch in branches:
            summary_path = _summary_path(config, model_name, seed, branch)
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            weights = payload.get("mixture_weights", {"m0": 1.0})
            for task, result in payload["mteb"].items():
                records.append({"seed": seed, "branch": branch, "weights": json.dumps(weights, sort_keys=True), "task": task, "score": result["score"]})
            records.append({"seed": seed, "branch": branch, "weights": json.dumps(weights, sort_keys=True), "task": "AGNewsClassification", "score": payload["AGNewsClassification"]["accuracy"]})
            records.append({"seed": seed, "branch": branch, "weights": json.dumps(weights, sort_keys=True), "task": "MSMARCORerankingGate", "score": payload["MSMARCORerankingGate"]["ndcg_at_10"]})
            for task, result in payload.get("CleanSTS", {}).items():
                records.append({"seed": seed, "branch": branch, "weights": json.dumps(weights, sort_keys=True), "task": task, "score": result["cosine_spearman"]})

        run_root = ROOT / "runs" / config["experiment_id"] / "full" / model_name / f"seed_{seed}"
        m0_geometry = run_root / "m0_geometry.json"
        if m0_geometry.exists():
            geometry_rows.append({"seed": seed, "branch": "m0", **json.loads(m0_geometry.read_text())})
        for spec in specs:
            task_path = run_root / spec["name"] / "task_geometry.json"
            trajectory_path = run_root / spec["name"] / "geometry_trajectory.csv"
            if not task_path.exists() or not trajectory_path.exists():
                continue
            with trajectory_path.open(encoding="utf-8") as handle:
                final = list(csv.DictReader(handle))[-1]
            geometry_rows.append(
                {
                    "seed": seed,
                    "branch": spec["name"],
                    **{key: float(value) for key, value in final.items() if key not in {"step", "fraction"}},
                    **json.loads(task_path.read_text()),
                }
            )

    if not records:
        raise RuntimeError("No mixture evaluations found")
    report_dir = ROOT / "reports" / "pilot_v2_mixtures"
    report_dir.mkdir(parents=True, exist_ok=True)
    scores = report_dir / "scores.csv"
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if geometry_rows:
        fields = ["seed", "branch", *sorted(set().union(*(row.keys() for row in geometry_rows)) - {"seed", "branch"})]
        with (report_dir / "geometry.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(geometry_rows)

    baseline = {(row["seed"], row["task"]): row["score"] for row in records if row["branch"] == "m0"}
    deltas = [{**row, "delta_vs_m0": row["score"] - baseline[(row["seed"], row["task"])]} for row in records if (row["seed"], row["task"]) in baseline]
    with (report_dir / "score_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*records[0], "delta_vs_m0"])
        writer.writeheader()
        writer.writerows(deltas)

    family_rows = []
    for seed in config["seeds"]:
        for branch in [spec["name"] for spec in specs]:
            branch_rows = [row for row in deltas if row["seed"] == seed and row["branch"] == branch]
            for family, tasks in FAMILIES.items():
                values = [row["delta_vs_m0"] for row in branch_rows if row["task"] in tasks]
                if len(values) == len(tasks):
                    family_rows.append({"seed": seed, "branch": branch, "family": family, "delta": float(np.mean(values))})
    with (report_dir / "family_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "branch", "family", "delta"])
        writer.writeheader()
        writer.writerows(family_rows)

    specs_by_name = {spec["name"]: spec for spec in specs}
    pure_branches = {
        "sts": "mix_s100_c000_r000",
        "classification": "mix_s000_c100_r000",
        "retrieval": "mix_s000_c000_r100",
    }
    interaction_rows = []
    for seed in config["seeds"]:
        for family in FAMILIES:
            pure = {
                objective: next(
                    row["delta"]
                    for row in family_rows
                    if row["seed"] == seed and row["branch"] == branch and row["family"] == family
                )
                for objective, branch in pure_branches.items()
            }
            for branch, spec in specs_by_name.items():
                if branch == "control":
                    continue
                observed_rows = [
                    row["delta"]
                    for row in family_rows
                    if row["seed"] == seed and row["branch"] == branch and row["family"] == family
                ]
                if not observed_rows:
                    continue
                expected = sum(spec["weights"][objective] * pure[objective] for objective in OBJECTIVES)
                interaction_rows.append(
                    {
                        "seed": seed,
                        "branch": branch,
                        "family": family,
                        "observed_delta": observed_rows[0],
                        "expected_linear_delta": expected,
                        "interaction_residual": observed_rows[0] - expected,
                    }
                )
    with (report_dir / "interaction_residuals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interaction_rows[0]))
        writer.writeheader()
        writer.writerows(interaction_rows)

    delta_frame = pd.DataFrame(deltas)
    family_frame = pd.DataFrame(family_rows)
    direct_rows = []
    for seed in config["seeds"]:
        for branch in specs_by_name:
            if branch == "control":
                continue
            direct_rows.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "sts": float(family_frame.query("seed == @seed and branch == @branch and family == 'sts'")["delta"].iloc[0]),
                    "classification": float(delta_frame.query("seed == @seed and branch == @branch and task == 'AGNewsClassification'")["delta_vs_m0"].iloc[0]),
                    "retrieval": float(family_frame.query("seed == @seed and branch == @branch and family == 'retrieval'")["delta"].iloc[0]),
                }
            )
    direct = pd.DataFrame(direct_rows)
    means = direct.groupby("branch")[[*OBJECTIVES]].mean()

    def nondominated(values: pd.DataFrame) -> set[str]:
        result = set()
        array = values.to_numpy()
        for index, name in enumerate(values.index):
            dominated = any(
                np.all(array[other] >= array[index]) and np.any(array[other] > array[index])
                for other in range(len(array))
                if other != index
            )
            if not dominated:
                result.add(name)
        return result

    primary_front = nondominated(means)
    rng = np.random.default_rng(20260921)
    bootstrap_counts = Counter()
    bootstrap_replicates = 10_000
    seeds = np.asarray(config["seeds"])
    for _ in range(bootstrap_replicates):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        rows = []
        for branch in means.index:
            subset = direct[direct.branch == branch].set_index("seed")
            rows.append([float(subset.loc[sampled, objective].mean()) for objective in OBJECTIVES])
        frame = pd.DataFrame(rows, index=means.index, columns=OBJECTIVES)
        bootstrap_counts.update(nondominated(frame))
    pareto_rows = []
    for branch, values in means.iterrows():
        pareto_rows.append(
            {
                "branch": branch,
                "sts_delta": values["sts"],
                "ag_news_delta": values["classification"],
                "retrieval_delta": values["retrieval"],
                "pareto_mean": branch in primary_front,
                "pareto_bootstrap_frequency": bootstrap_counts[branch] / bootstrap_replicates,
            }
        )
    pd.DataFrame(pareto_rows).to_csv(report_dir / "pareto_points.csv", index=False)

    criteria_rows = []
    broad_families = {"sts", "classification_transfer", "clustering", "retrieval"}
    direct_mean = direct.groupby("branch")[[*OBJECTIVES]].mean()
    family_mean = family_frame.groupby(["branch", "family"])["delta"].mean()
    for branch, spec in specs_by_name.items():
        if branch == "control" or branch in pure_branches.values():
            continue
        active = [objective for objective in OBJECTIVES if spec["weights"][objective] > 0]
        all_retained = True
        all_attenuated = True
        for objective in active:
            pure_branch = pure_branches[objective]
            pure_gain = float(direct_mean.loc[pure_branch, objective])
            observed_gain = float(direct_mean.loc[branch, objective])
            retention = observed_gain / pure_gain if pure_gain > 0 else math.nan
            non_target = broad_families - {objective}
            if objective == "classification":
                non_target = broad_families
            pure_worst = min(float(family_mean.loc[(pure_branch, family)]) for family in non_target)
            mixture_worst = min(float(family_mean.loc[(branch, family)]) for family in non_target)
            retained = bool(pure_gain > 0 and retention >= 0.5)
            attenuated = bool(mixture_worst - pure_worst >= 0.01)
            all_retained &= retained
            all_attenuated &= attenuated
            criteria_rows.append(
                {
                    "branch": branch,
                    "objective": objective,
                    "pure_target_gain": pure_gain,
                    "mixture_target_gain": observed_gain,
                    "retained_fraction": retention,
                    "pure_worst_non_target": pure_worst,
                    "mixture_worst_non_target": mixture_worst,
                    "worst_delta_improvement": mixture_worst - pure_worst,
                    "retains_50_percent": retained,
                    "attenuates_by_0_01": attenuated,
                }
            )
        for row in criteria_rows:
            if row["branch"] == branch:
                row["mixture_meets_attenuation_rule"] = all_retained and all_attenuated
        family_values = [float(family_mean.loc[(branch, family)]) for family in broad_families]
        for row in criteria_rows:
            if row["branch"] == branch:
                row["all_clean_families_nonnegative"] = min(family_values) >= 0
    pd.DataFrame(criteria_rows).to_csv(report_dir / "attenuation_criteria.csv", index=False)

    lines = [
        "# Pilot v2: objective mixtures",
        "",
        f"Completed evaluation summaries: {len(set((row['seed'], row['branch']) for row in records))} / {len(config['seeds']) * (len(specs) + 1)}.",
        "",
        "| Branch | STS | Classification | Classification transfer | Clustering | Retrieval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch in [spec["name"] for spec in specs]:
        values = {}
        for family in FAMILIES:
            selected = [row["delta"] for row in family_rows if row["branch"] == branch and row["family"] == family]
            values[family] = float(np.mean(selected)) if selected else math.nan
        lines.append(
            f"| {branch} | {values['sts']:+.4f} | {values['classification']:+.4f} | {values['classification_transfer']:+.4f} | {values['clustering']:+.4f} | {values['retrieval']:+.4f} |"
        )
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

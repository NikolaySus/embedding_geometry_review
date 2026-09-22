from __future__ import annotations

import json
import math
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .common import ROOT, atomic_directory, check_disk, device, read_jsonl, set_seed, write_json
from .geometry import append_snapshot, generic_snapshot, task_geometry
from .modeling import encode_features, load_encoder, verify_normalized


BRANCHES = ("control", "sts", "classification", "retrieval")


def _batches(rows: list[dict[str, Any]], batch_size: int, epochs: int, seed: int, branch: str):
    rng = random.Random(seed)
    if branch != "classification":
        for epoch in range(epochs):
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                if len(selected) == batch_size:
                    yield [rows[index] for index in selected]
        return

    grouped = {label: [row for row in rows if int(row["label"]) == label] for label in range(4)}
    per_class = batch_size // 4
    for _ in range(epochs):
        for values in grouped.values():
            rng.shuffle(values)
        offsets = {label: 0 for label in grouped}
        for _ in range(len(rows) // batch_size):
            batch = []
            for label, values in grouped.items():
                start = offsets[label]
                batch.extend(values[start : start + per_class])
                offsets[label] += per_class
            rng.shuffle(batch)
            yield batch


def _simcse_loss(model, batch, torch_device, temperature: float):
    texts = [row["text"] for row in batch]
    first = encode_features(model, texts, torch_device)
    second = encode_features(model, texts, torch_device)
    logits = first @ second.T / temperature
    labels = torch.arange(len(batch), device=torch_device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def _cosent_loss(model, batch, torch_device, scale: float):
    first = encode_features(model, [row["sentence1"] for row in batch], torch_device)
    second = encode_features(model, [row["sentence2"] for row in batch], torch_device)
    similarities = (first * second).sum(dim=1) * scale
    labels = torch.tensor([float(row["score"]) for row in batch], device=torch_device)
    differences = similarities[:, None] - similarities[None, :]
    mask = labels[:, None] < labels[None, :]
    selected = differences[mask]
    return torch.logsumexp(torch.cat([torch.zeros(1, device=torch_device), selected]), dim=0)


def _supervised_contrastive_loss(model, batch, torch_device, temperature: float):
    embeddings = encode_features(model, [row["text"] for row in batch], torch_device)
    labels = torch.tensor([int(row["label"]) for row in batch], device=torch_device)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(batch), dtype=torch.bool, device=torch_device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits) * ~self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    return -(positive_mask * log_prob).sum(dim=1).div(positive_mask.sum(dim=1).clamp_min(1)).mean()


def _retrieval_loss(model, batch, torch_device, temperature: float):
    queries = encode_features(model, [row["query"] for row in batch], torch_device)
    positives = encode_features(model, [row["positive"] for row in batch], torch_device)
    negatives = encode_features(model, [row["negative"] for row in batch], torch_device)
    candidates = torch.cat([positives, negatives], dim=0)
    logits = queries @ candidates.T / temperature
    labels = torch.arange(len(batch), device=torch_device)
    return F.cross_entropy(logits, labels)


def _schedule(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, round(total_steps * warmup_ratio))

    def factor(step: int):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return LambdaLR(optimizer, factor)


def _save_model(model, target: Path) -> None:
    temporary = atomic_directory(target)
    model.save_pretrained(str(temporary), safe_serialization=True)
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)


def _train_loop(
    model,
    rows,
    *,
    branch: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    reference_path: Path | None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    diagnostics_config = config
    if max_steps is not None:
        diagnostics_config = deepcopy(config)
        diagnostics_config["geometry"]["generic_sample_size"] = min(
            512, config["geometry"]["generic_sample_size"]
        )
        diagnostics_config["geometry"]["pair_sample_size"] = min(
            5_000, config["geometry"]["pair_sample_size"]
        )
    torch_device = device()
    model.to(torch_device)
    model.train()
    batches = list(_batches(rows, batch_size, epochs, seed, branch))
    if max_steps is not None:
        batches = batches[:max_steps]
    total_steps = len(batches)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config["training"]["weight_decay"])
    scheduler = _schedule(optimizer, total_steps, config["training"]["warmup_ratio"])
    fractions = (0.0, 1.0) if max_steps is not None else config["training"]["trajectory_fractions"]
    snapshot_steps = {round(fraction * total_steps): fraction for fraction in fractions}
    metrics_path = output_dir / "train_metrics.jsonl"
    geometry_path = output_dir / "geometry_trajectory.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    losses = []

    def snapshot(step: int, fraction: float):
        model.eval()
        metrics = generic_snapshot(model, diagnostics_config, seed=seed, reference_path=reference_path)
        append_snapshot(geometry_path, {"step": step, "fraction": fraction, **metrics})
        model.train()

    if 0 in snapshot_steps:
        snapshot(0, snapshot_steps[0])

    with metrics_path.open("w", encoding="utf-8") as log:
        for step, batch in enumerate(tqdm(batches, desc=f"{branch} seed={seed}"), start=1):
            optimizer.zero_grad(set_to_none=True)
            dtype = torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float32
            with torch.autocast(device_type=torch_device.type, dtype=dtype, enabled=torch_device.type == "cuda"):
                if branch == "control":
                    loss = _simcse_loss(model, batch, torch_device, config["training"]["temperature"])
                elif branch == "sts":
                    loss = _cosent_loss(model, batch, torch_device, config["training"]["cosent_scale"])
                elif branch == "classification":
                    loss = _supervised_contrastive_loss(model, batch, torch_device, config["training"]["temperature"])
                else:
                    loss = _retrieval_loss(model, batch, torch_device, config["training"]["temperature"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            value = float(loss.detach())
            losses.append(value)
            record = {"step": step, "loss": value, "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0]}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step in snapshot_steps:
                snapshot(step, snapshot_steps[step])

    model.eval()
    full_geometry = task_geometry(model, diagnostics_config, seed)
    write_json(output_dir / "task_geometry.json", full_geometry)
    result = {
        "branch": branch,
        "seed": seed,
        "steps": total_steps,
        "examples": total_steps * batch_size,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "elapsed_seconds": time.time() - started,
        "embedding_norms": verify_normalized(model, ["A short diagnostic sentence.", "A different diagnostic sentence."]),
        "technical_status": "pass",
    }
    write_json(output_dir / "manifest.json", result)
    _save_model(model, output_dir / "final_model")
    return result


def build_raw_m0(config: dict[str, Any], model_name: str, seed: int, smoke: bool = False) -> Path:
    model_cfg = config["models"][model_name]
    mode = "smoke" if smoke else "full"
    output_dir = ROOT / "runs" / config["experiment_id"] / mode / model_name / f"seed_{seed}" / "m0"
    final_model = output_dir / "final_model"
    if (output_dir / "manifest.json").exists() and final_model.exists():
        return final_model
    rows = read_jsonl("data/processed/neutral_m0.jsonl")
    model = load_encoder(
        model_cfg["model_id"],
        max_seq_length=model_cfg["max_seq_length"],
        device=device(),
        raw_backbone=True,
    )
    result = _train_loop(
        model,
        rows,
        branch="control",
        epochs=1,
        batch_size=config["training"]["m0_batch_size"],
        learning_rate=config["training"]["m0_learning_rate"],
        config=config,
        seed=seed,
        output_dir=output_dir,
        reference_path=None,
        max_steps=config["training"]["smoke_steps"] if smoke else None,
    )
    result["role"] = "neutral_m0"
    write_json(output_dir / "manifest.json", result)
    return final_model


def train_model_seed(config: dict[str, Any], model_name: str, seed: int, smoke: bool = False) -> dict[str, Any]:
    check_disk(config["storage"]["minimum_free_gb"])
    model_cfg = config["models"][model_name]
    mode = "smoke" if smoke else "full"
    run_root = ROOT / "runs" / config["experiment_id"] / mode / model_name / f"seed_{seed}"
    if model_cfg["build_neutral_m0"]:
        source = build_raw_m0(config, model_name, seed, smoke=smoke)
        source_is_raw = False
    else:
        source = model_cfg["model_id"]
        source_is_raw = False

    baseline_model = load_encoder(
        source,
        max_seq_length=model_cfg["max_seq_length"],
        device=device(),
        raw_backbone=source_is_raw,
    )
    reference_path = run_root / "m0_reference_embeddings.npy"
    baseline_geometry = generic_snapshot(
        baseline_model,
        config,
        seed=seed,
        save_reference_path=reference_path,
    )
    write_json(run_root / "m0_geometry.json", {**baseline_geometry, **task_geometry(baseline_model, config, seed)})
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    datasets = {
        "control": read_jsonl("data/processed/neutral_control.jsonl"),
        "sts": read_jsonl("data/processed/sts.jsonl"),
        "classification": read_jsonl("data/processed/classification.jsonl"),
        "retrieval": read_jsonl("data/processed/retrieval.jsonl"),
    }
    results = {}
    for branch in BRANCHES:
        output_dir = run_root / branch
        if (output_dir / "manifest.json").exists() and (output_dir / "final_model").exists():
            results[branch] = json.loads((output_dir / "manifest.json").read_text())
            continue
        model = load_encoder(
            source,
            max_seq_length=model_cfg["max_seq_length"],
            device=device(),
            raw_backbone=False,
        )
        results[branch] = _train_loop(
            model,
            datasets[branch],
            branch=branch,
            epochs=config["data"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            config=config,
            seed=seed,
            output_dir=output_dir,
            reference_path=reference_path,
            max_steps=config["training"]["smoke_steps"] if smoke else None,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

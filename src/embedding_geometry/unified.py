"""One training executor for pure objectives, mixtures and resumable diagnostics."""

from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
import torch
from torch.optim import AdamW

from .common import ROOT, check_disk, read_jsonl, set_seed
from .mixtures import mixture_specs, objective_schedule, _batch_stream, _loss
from .training import _schedule, _save_model
from .unified_evaluation import evaluate
from .unified_geometry import VERSION, encode, probe_texts, snapshot
from .unified_state import (atomic_json, atomic_torch, bind, digest, isolated,
                            model_digest, provenance, rng_state, restore_rng, result_root)


SMOKE_REGIMES = ("control", "mix_s100_c000_r000", "mix_s000_c100_r000", "mix_s000_c000_r100", "mix_s033_c033_r033")


def validate_config(config):
    if config["schema_version"] != 1 or config["geometry"]["version"] != VERSION:
        raise ValueError("Unsupported schema or geometry definition")
    for model in config["models"].values():
        if len(model["revision"]) != 40:
            raise ValueError("Use an immutable 40-character model revision")
    if config["training"]["batch_size"] % 4:
        raise ValueError("Balanced four-class batches require batch_size divisible by 4")
    if config["training"]["snapshot_steps"][-1] != config["training"]["total_steps"]:
        raise ValueError("Final snapshot must equal total_steps")
    if not set(config["evaluation"]["gate_tasks"]) <= set(config["evaluation"]["full_tasks"]):
        raise ValueError("Gate tasks must be a subset of full tasks")


def plan(config):
    validate_config(config)
    specs = mixture_specs(config)
    models, seeds = len(config["models"]), len(config["seeds"])
    runs = len(specs)*models*seeds
    intermediate = len(config["evaluation"]["trajectory_regimes"])*2*models*seeds
    return {"experiment_id": config["experiment_id"], "training_runs": runs,
            "steps_per_run": config["training"]["total_steps"],
            "total_training_steps": runs*config["training"]["total_steps"],
            "intermediate_gates": intermediate, "final_full_evaluations": runs,
            "baseline_full_evaluations": models, "geometry_snapshots": runs*4,
            "retention": "All final models; latest optimizer checkpoints retained for recovery",
            "estimated_storage_gib": round(runs*.5 + 5, 1),
            "storage_assumption": "MiniLM weights ~90MB, AdamW+RNG ~360MB per run; caches extra",
            "regimes": specs}


def context(config, smoke=False):
    validate_config(config)
    configure_runtime(config)
    check_disk(config["storage"]["minimum_free_gb"])
    identity = provenance(config)
    root = result_root(config, smoke, identity["identity"])
    bind(root / "protocol.json", identity)
    return root, identity["identity"]


def configure_runtime(config):
    if config["training"].get("deterministic", True):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_model(config, model_name):
    cfg = config["models"][model_name]
    # Loading by snapshot path also pins nested tokenizer/module lookups.
    files = ["*.json", "*.safetensors", "*.txt", "*.model"]
    try:
        source = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True, allow_patterns=files)
    except LocalEntryNotFoundError:
        source = snapshot_download(cfg["model_id"], revision=cfg["revision"], allow_patterns=files)
    model = SentenceTransformer(source, device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"attn_implementation": cfg.get("attention_implementation", "eager")})
    model.max_seq_length = cfg["max_seq_length"]
    return model


def baseline(config, root, protocol, model_name, smoke=False):
    folder = root / model_name / "m0"
    with isolated():
        model = load_model(config, model_name)
        identity = {"protocol": protocol, "model": model_name, "branch": "m0", "seed": None,
                    "step": 0, "weights_sha256": model_digest(model)}
        bind(folder / "identity.json", identity)
        texts, ids = probe_texts(config, smoke)
        reference_file = folder / "reference.npy"
        if not reference_file.exists():
            reference_file.parent.mkdir(parents=True, exist_ok=True)
            temp = reference_file.with_suffix(".tmp")
            with temp.open("wb") as handle:
                np.save(handle, encode(model, texts, config["evaluation"]["batch_size"]))
            temp.replace(reference_file)
        reference = np.load(reference_file)
        if not (folder / "geometry.json").exists():
            with isolated(model):
                metrics = snapshot(model, config, reference, ids, smoke)
            atomic_json(folder / "geometry.json", {"identity": identity, **metrics})
        evaluate(model, config, folder / "evaluation", identity, "full", root / "datasets", smoke)
        del model
    return reference, ids, identity


def token_counts(model, objective, batch):
    keys = {"control": ("text",), "classification": ("text",), "sts": ("sentence1", "sentence2"),
            "retrieval": ("query", "positive", "negative")}[objective]
    counts = []
    for key in keys:
        features = model.tokenize([row[key] for row in batch])
        counts.append(int(features["attention_mask"].sum()))
    multiplier = 2 if objective == "control" else 1
    return sum(counts)*multiplier


def train(config, root, protocol, model_name, seed, spec, reference, ids, smoke=False, stop_after=None):
    configure_runtime(config)
    name, cfg = spec["name"], config["training"]
    output = root / model_name / f"seed_{seed}" / name
    identity = {"protocol": protocol, "model": model_name, "branch": name, "seed": seed}
    bind(output / "identity.json", identity)
    schedule = objective_schedule(spec["weights"], cfg["smoke_steps"] if smoke else cfg["total_steps"], seed)
    total = len(schedule)
    checkpoints = {0, total//2, total} if smoke else set(cfg["snapshot_steps"])
    set_seed(seed)
    model = load_model(config, model_name)
    model.train()
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = _schedule(optimizer, total, cfg["warmup_ratio"])
    files = {"control": "neutral_control", "sts": "sts", "classification": "classification", "retrieval": "retrieval"}
    streams = {key: _batch_stream(read_jsonl(f"data/processed/{files[key]}.jsonl"), cfg["batch_size"], seed, key)
               for key in sorted(set(schedule))}
    state_file = output / "checkpoints/latest.pt"
    records, step = [], 0
    if state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        if state["identity"] != identity or state["schedule"] != schedule:
            raise ValueError("Cannot resume an incompatible checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        records, step = state["records"], state["step"]
        for objective, count in Counter(schedule[:step]).items():
            for _ in range(count):
                next(streams[objective])
        restore_rng(state["rng"])
        atomic_json(output / "training.json", records)

    def save():
        check_disk(config["storage"]["minimum_free_gb"])
        atomic_torch(state_file, {"identity": identity, "schedule": schedule, "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(), "records": records, "stream_counts": dict(Counter(schedule[:step]))})
        atomic_json(output / "training.json", records)

    def measure():
        folder = output / "observations" / f"step_{step:04d}"
        with isolated(model):
            measured = {**identity, "step": step, "weights_sha256": model_digest(model)}
            bind(folder / "identity.json", measured)
            path = folder / "geometry.json"
            if not path.exists():
                start = time.monotonic()
                values = snapshot(model, config, reference, ids, smoke)
                atomic_json(path, {"identity": measured, **values, "seconds": time.monotonic()-start})
            if step == total or (step > 0 and (smoke or name in config["evaluation"]["trajectory_regimes"])):
                suite = "full" if step == total else "gate"
                evaluate(model, config, folder / "evaluation", measured, suite, root / "datasets", smoke)

    if step in checkpoints:
        save()
        measure()
    for index in range(step, total):
        start = time.monotonic()
        objective = schedule[index]
        batch = next(streams[objective])
        if len(batch) != cfg["batch_size"]:
            raise ValueError("Unexpected incomplete training batch")
        tokens = token_counts(model, objective, batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and cfg["mixed_precision"] == "bf16"):
            loss = _loss(model, objective, batch, device, config)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {name}, step {index+1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step = index+1
        records.append({"step": step, "objective": objective, "loss": float(loss.detach()),
                        "grad_norm": float(norm), "lr": scheduler.get_last_lr()[0], "examples": len(batch),
                        "forward_tokens": tokens, "seconds": time.monotonic()-start})
        atomic_json(output / "training.json", records)
        print(f"{model_name} {name} seed={seed} {step}/{total} loss={records[-1]['loss']:.4f}", flush=True)
        if step % cfg["checkpoint_every"] == 0 or step in checkpoints or step == stop_after:
            save()
        if step in checkpoints:
            measure()
        if step == stop_after:
            return {"status": "interrupted_for_test", "step": step}
    _save_model(model, output / "final_model")
    result = {**identity, "status": "complete", "technical_only": smoke, "weights": spec["weights"],
              "step_counts": dict(Counter(schedule)), "schedule_sha256": digest(schedule), "steps": total,
              "weights_sha256": model_digest(model), "training_seconds": sum(r["seconds"] for r in records),
              "examples": sum(r["examples"] for r in records), "forward_tokens": sum(r["forward_tokens"] for r in records)}
    atomic_json(output / "manifest.json", result)
    return result


def run(config, *, smoke=False, models=None, seeds=None, regimes=None):
    root, protocol = context(config, smoke)
    selected_models = models or list(config["models"])
    selected_seeds = seeds or (config["seeds"][:1] if smoke else config["seeds"])
    selected_regimes = regimes or (SMOKE_REGIMES if smoke else [s["name"] for s in mixture_specs(config)])
    specs = {s["name"]: s for s in mixture_specs(config)}
    if set(selected_models)-set(config["models"]) or set(selected_seeds)-set(config["seeds"]) or set(selected_regimes)-set(specs):
        raise ValueError("Unknown selected model, seed or regime")
    remaining = sum(not (root / model / f"seed_{seed}" / name / "manifest.json").exists()
                    for model in selected_models for seed in selected_seeds for name in selected_regimes)
    check_disk(config["storage"]["minimum_free_gb"] + remaining*.5)
    results = []
    for model_name in selected_models:
        reference, ids, _ = baseline(config, root, protocol, model_name, smoke)
        for seed in selected_seeds:
            for name in selected_regimes:
                results.append(train(config, root, protocol, model_name, seed, specs[name], reference, ids, smoke))
    atomic_json(root / "last_invocation.json", results)
    return results


def evaluate_saved(config, *, models=None, seeds=None, regimes=None):
    root, protocol = context(config)
    selected = set(regimes or [s["name"] for s in mixture_specs(config)])
    count = 0
    for model_name in models or config["models"]:
        baseline(config, root, protocol, model_name)
        for seed in seeds or config["seeds"]:
            for name in sorted(selected):
                output = root / model_name / f"seed_{seed}" / name
                checkpoint = output / "checkpoints/latest.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if state["identity"]["protocol"] != protocol:
                    raise ValueError("Checkpoint protocol mismatch")
                model = load_model(config, model_name)
                model.load_state_dict(state["model"])
                step = state["step"]
                if step not in config["training"]["snapshot_steps"] or step == 0:
                    raise ValueError("Latest checkpoint is not an evaluation point; resume training with run")
                identity = {**state["identity"], "step": step, "weights_sha256": model_digest(model)}
                suite = "full" if step == config["training"]["total_steps"] else "gate"
                evaluate(model, config, output / "observations" / f"step_{step:04d}" / "evaluation",
                         identity, suite, root / "datasets")
                del model
                count += 1
    return {"evaluated_checkpoints": count}

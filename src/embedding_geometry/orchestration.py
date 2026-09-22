from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import ROOT, check_disk, write_json
from .evaluation import evaluate_checkpoint
from .training import BRANCHES, train_model_seed


def _run_root(config: dict[str, Any], model_name: str, seed: int, smoke: bool = False) -> Path:
    mode = "smoke" if smoke else "full"
    return ROOT / "runs" / config["experiment_id"] / mode / model_name / f"seed_{seed}"


def validate_technical(config: dict[str, Any], *, smoke: bool, seeds: list[int]) -> dict[str, Any]:
    checks = []
    for model_name in config["models"]:
        for seed in seeds:
            root = _run_root(config, model_name, seed, smoke)
            if config["models"][model_name]["build_neutral_m0"]:
                manifest = root / "m0" / "manifest.json"
                checks.append({"path": str(manifest.relative_to(ROOT)), "exists": manifest.exists()})
            for branch in BRANCHES:
                manifest = root / branch / "manifest.json"
                passed = False
                detail = None
                if manifest.exists():
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    norms = payload["embedding_norms"]
                    passed = (
                        payload.get("technical_status") == "pass"
                        and payload["steps"] > 0
                        and 0.98 <= norms["minimum"] <= 1.02
                        and 0.98 <= norms["maximum"] <= 1.02
                    )
                    detail = payload
                checks.append({"path": str(manifest.relative_to(ROOT)), "exists": manifest.exists(), "passed": passed, "detail": detail})
    passed = all(item.get("exists", False) and item.get("passed", True) for item in checks)
    result = {"passed": passed, "smoke": smoke, "seeds": seeds, "checks": checks}
    write_json(ROOT / "reports" / f"pilot_v1_technical_{'smoke' if smoke else 'full'}.json", result)
    if not passed:
        raise RuntimeError("Technical validation failed; see the generated technical report")
    return result


def train_phase(config: dict[str, Any], phase: str) -> None:
    check_disk(config["storage"]["minimum_free_gb"])
    if phase == "smoke":
        for model_name in config["models"]:
            train_model_seed(config, model_name, config["seeds"][0], smoke=True)
        validate_technical(config, smoke=True, seeds=[config["seeds"][0]])
        return

    seeds = [config["seeds"][0]] if phase == "seed1" else config["seeds"]
    for seed in seeds:
        for model_name in config["models"]:
            train_model_seed(config, model_name, seed, smoke=False)
    validate_technical(config, smoke=False, seeds=seeds)


def checkpoint_for(config: dict[str, Any], model_name: str, seed: int, branch: str) -> str | Path:
    if branch == "m0":
        if config["models"][model_name]["build_neutral_m0"]:
            return _run_root(config, model_name, seed) / "m0" / "final_model"
        return config["models"][model_name]["model_id"]
    return _run_root(config, model_name, seed) / branch / "final_model"


def evaluate_phase(config: dict[str, Any], suite: str, seeds: list[int] | None = None) -> None:
    seeds = seeds or config["seeds"]
    for seed in seeds:
        for model_name in config["models"]:
            for branch in ("m0", *BRANCHES):
                output = ROOT / "runs" / config["experiment_id"] / "mteb" / suite / model_name / f"seed_{seed}" / branch / "evaluation_summary.json"
                if output.exists():
                    continue
                evaluate_checkpoint(
                    config,
                    checkpoint_for(config, model_name, seed, branch),
                    model_name=model_name,
                    seed=seed,
                    branch=branch,
                    suite=suite,
                )


def run_all(config: dict[str, Any]) -> None:
    train_phase(config, "smoke")
    train_phase(config, "seed1")
    evaluate_phase(config, "gate", [config["seeds"][0]])
    train_phase(config, "seeds3")
    evaluate_phase(config, "gate", config["seeds"])
    evaluate_phase(config, "full", config["seeds"])


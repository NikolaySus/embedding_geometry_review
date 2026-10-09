"""Versioned identities, isolated randomness and atomic resumable state."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .common import ROOT, file_sha256


DATA_FILES = ("neutral_control", "sts", "classification", "retrieval", "neutral_diagnostic", "retrieval_gate")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def isolated(model=None):
    state = rng_state()
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    try:
        if model is not None:
            model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(state)


def provenance(config):
    data = {name: file_sha256(ROOT / "data/processed" / f"{name}.jsonl") for name in DATA_FILES}
    audits = {str(p.relative_to(ROOT)): file_sha256(p)
              for p in sorted((ROOT / "reports").rglob("*audit*.json"))}
    manifest = ROOT / "data/processed/manifest.json"
    if manifest.exists():
        audits[str(manifest.relative_to(ROOT))] = file_sha256(manifest)
    code = {str(p.relative_to(ROOT)): file_sha256(p)
            for p in sorted((ROOT / "src/embedding_geometry").glob("*.py"))}
    payload = {"schema_version": 1, "config": config, "data": data, "audits": audits,
               "code": code, "uv_lock": file_sha256(ROOT / "uv.lock")}
    return {**payload, "identity": digest(payload)}


def result_root(config, smoke=False, identity=None):
    root = ROOT / "runs" / config["experiment_id"]
    if smoke:
        identity = identity or provenance(config)["identity"]
        return root / "smoke" / identity[:12]
    return root / "full"


def bind(path, identity):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"Incompatible run/cache identity: {path}; use a new experiment ID")
    else:
        atomic_json(path, identity)


def model_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        array = tensor.detach().cpu().contiguous()
        result.update(str(array.dtype).encode())
        result.update(str(tuple(array.shape)).encode())
        result.update(array.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()

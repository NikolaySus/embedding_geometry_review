"""Run five technical GPU branches and verify a mixed-objective resume."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from embedding_geometry.common import ROOT, load_config
from embedding_geometry.mixtures import mixture_specs
from embedding_geometry.unified import run, context, train
from embedding_geometry.unified_geometry import probe_texts
from embedding_geometry.unified_report import report
from embedding_geometry.unified_state import atomic_json


def main():
    config = load_config("configs/unified_v1.json")
    start = time.monotonic()
    results = run(config, smoke=True)
    root, protocol = context(config, True)
    model, seed = "all_minilm", 42
    spec = next(s for s in mixture_specs(config) if s["name"] == "mix_s033_c033_r033")
    _, ids = probe_texts(config, True)
    reference = np.load(root / model / "m0/reference.npy")
    resume_root = ROOT / "runs" / config["experiment_id"] / "smoke_verification" / protocol[:12]
    first = train(config, resume_root, protocol, model, seed, spec, reference, ids, True, stop_after=10)
    resumed = train(config, resume_root, protocol, model, seed, spec, reference, ids, True)
    continuous = next(row for row in results if row["branch"] == spec["name"])
    if resumed["weights_sha256"] != continuous["weights_sha256"]:
        raise AssertionError("GPU resumed weights differ from continuous smoke")
    summary = report(config, smoke=True)
    output = ROOT / "reports" / config["experiment_id"] / "smoke" / protocol[:12] / "validation.json"
    atomic_json(output, {"status": "passed", "technical_only": True, "protocol": protocol,
        "five_regimes": len(results), "steps_each": 20, "resume_stop": first["step"] if "step" in first else 20,
        "resume_weights_match": True, "elapsed_seconds": time.monotonic()-start,
        "report": summary, "training_seconds": sum(r["training_seconds"] for r in results)})
    print(output)


if __name__ == "__main__":
    main()

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from embedding_geometry.common import ROOT, load_config
from embedding_geometry.unified import plan
from embedding_geometry.unified_geometry import neighbors, preservation, global_metrics
from embedding_geometry.unified_state import atomic_json, bind, isolated, rng_state, restore_rng
from embedding_geometry.unified_report import family_values


def test_design():
    cfg = load_config("configs/unified_v1.json")
    result = plan(cfg)
    assert result["training_runs"] == 69
    assert result["intermediate_gates"] == 36
    assert result["total_training_steps"] == 18630
    assert len({s["name"] for s in result["regimes"]}) == 23
    from embedding_geometry.mixtures import objective_schedule
    for spec in result["regimes"]:
        assert len(objective_schedule(spec["weights"], 270, 42)) == 270


def test_true_first_neighbors():
    angles = np.array([0, .01, .1, .4])
    x = np.column_stack([np.cos(angles), np.sin(angles)])
    actual = neighbors(x, 1)
    assert actual[0, 0] == 1
    assert actual[1, 0] == 0
    assert not any(i in row for i, row in enumerate(actual))
    duplicates = np.ones((5, 3))
    assert neighbors(duplicates, 4).shape == (5, 4)
    assert not any(i in row for i, row in enumerate(neighbors(duplicates, 4)))
    assert preservation(x, x, 2, list(range(4)), list(range(4))) == 1
    with pytest.raises(ValueError):
        preservation(x, x, 2, [1, 2, 3, 4], [2, 1, 3, 4])
    with pytest.raises(ValueError):
        neighbors(x, 4)


def test_geometry_spectrum():
    x = np.random.default_rng(1).normal(size=(30, 5))
    x /= np.linalg.norm(x, axis=1)[:, None]
    result, spectrum = global_metrics(x, 10, 100, 42)
    assert len(spectrum) == 5
    assert 1 <= result["effective_rank"] <= 5
    assert result["top10_eigenvalue_share"] == pytest.approx(1)
    with pytest.raises(ValueError):
        global_metrics(np.ones((30, 5)), 10, 100, 42)


def test_rng_and_model_mode_isolation():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout())
    model.train()
    model[0].eval()
    state = rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    with isolated(model):
        assert not model.training
        random.random(), np.random.random(), torch.rand(3)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    assert model.training and not model[0].training


def test_cache_identity_and_missing_families(tmp_path):
    bind(tmp_path / "cache.json", {"checkpoint": "a", "step": 68})
    bind(tmp_path / "cache.json", {"checkpoint": "a", "step": 68})
    with pytest.raises(ValueError):
        bind(tmp_path / "cache.json", {"checkpoint": "a", "step": 135})
    with pytest.raises(ValueError):
        family_values(pd.DataFrame({"task": ["a"], "score": [.5]}), ["a", "b"])
    with pytest.raises(ValueError):
        family_values(pd.DataFrame({"task": ["a", "a"], "score": [.5, .6]}), ["a"])


def test_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from embedding_geometry import unified as u

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
            self.drop = torch.nn.Dropout(.2)

    monkeypatch.setattr(u, "load_model", lambda *args: Tiny())
    monkeypatch.setattr(u, "check_disk", lambda *args: None)
    monkeypatch.setattr(u, "read_jsonl", lambda *args: [{"text": str(i)} for i in range(16)])
    monkeypatch.setattr(u, "token_counts", lambda *args: 10)
    monkeypatch.setattr(u, "_save_model", lambda model, path: None)

    def loss(model, objective, batch, device, config):
        x = torch.tensor([[float(row["text"]), 1., -1.] for row in batch])
        return model.layer(model.drop(x)).square().mean()

    def measurement(*args):
        random.random(), np.random.random(), torch.rand(5)
        return {"metrics": {"test": 1.}, "version": "test"}

    monkeypatch.setattr(u, "_loss", loss)
    monkeypatch.setattr(u, "snapshot", measurement)
    monkeypatch.setattr(u, "evaluate", measurement)
    config = load_config("configs/unified_v1.json")
    config["training"].update(total_steps=4, snapshot_steps=[0, 2, 4], batch_size=4, checkpoint_every=2)
    spec = {"name": "control", "weights": {"control": 1.}}
    args = (config, "protocol", "all_minilm", 42, spec, np.ones((12, 3)), list(range(12)))
    def run(root, stop=None):
        return u.train(args[0], root, *args[1:], stop_after=stop)
    one = run(tmp_path / "continuous")
    assert run(tmp_path / "resumed", 2)["step"] == 2
    two = run(tmp_path / "resumed")
    assert one["weights_sha256"] == two["weights_sha256"]
    rows = json.loads((tmp_path / "resumed/all_minilm/seed_42/control/training.json").read_text())
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    # Repeating a completed run does not add optimizer updates.
    assert run(tmp_path / "resumed")["weights_sha256"] == one["weights_sha256"]


def test_historical_adapter():
    from gallery.bundles import load_bundle
    from gallery.data import load_absolute_data, load_data
    bundle = load_bundle()
    pd.testing.assert_frame_equal(bundle["absolute"], load_absolute_data())
    pd.testing.assert_frame_equal(bundle["deltas"], load_data()[0])


def test_incomplete_gallery_is_blocked(tmp_path, monkeypatch):
    from gallery import bundles
    monkeypatch.setattr(bundles, "ROOT", tmp_path)
    atomic_json(tmp_path / "reports/partial/full/status.json", {"gallery_ready": False})
    with pytest.raises(ValueError, match="Incomplete"):
        bundles.load_bundle("partial")


def test_complete_report_and_linear_residuals(tmp_path, monkeypatch):
    from embedding_geometry.unified_report import FAMILIES, GEOMETRY, export_gallery
    from embedding_geometry.mixtures import mixture_specs
    config = load_config("configs/unified_v1.json")
    specs = mixture_specs(config)
    tasks = list(dict.fromkeys(task for values in FAMILIES.values() for task in values))
    scores, geometry = [], []
    for seed in config["seeds"]:
        for spec in specs:
            delta = .01*spec["weights"].get("sts", 0)-.02*spec["weights"].get("classification", 0)
            for step in config["training"]["snapshot_steps"]:
                identity = {"branch": spec["name"], "seed": seed, "model": "all_minilm", "step": step}
                scores.extend({**identity, "task": task, "score": .5+delta*step/270} for task in tasks)
                geometry.append({**identity, **dict.fromkeys(GEOMETRY, .5+delta*step/270),
                                 "mean_cosine": .5, "geometry_version": "test", "probe_ids_hash": "same"})
    base = {"branch": "m0", "seed": None, "model": "all_minilm", "step": 0}
    scores.extend({**base, "task": task, "score": .5} for task in tasks)
    geometry.append({**base, **dict.fromkeys(GEOMETRY, .5), "mean_cosine": .5,
                     "geometry_version": "test", "probe_ids_hash": "same"})
    frames = {"scores": pd.DataFrame(scores), "geometry": pd.DataFrame(geometry)}
    destination = tmp_path / "reports" / config["experiment_id"] / "full/all_minilm"
    export_gallery(config, frames, specs, "all_minilm", destination)
    residual = pd.read_csv(destination / "residuals.csv")
    np.testing.assert_allclose(residual.interaction_residual, 0, atol=1e-15)
    assert len(pd.read_csv(destination / "absolute.csv")) == 72
    assert len(pd.read_csv(destination / "quality_trajectories.csv")) == 6*3*4*5
    assert len(pd.read_csv(destination / "pareto.csv")) == 66
    from gallery import bundles
    from pypdf import PdfReader
    monkeypatch.setattr(bundles, "ROOT", tmp_path)
    atomic_json(destination.parent / "status.json", {"gallery_ready": True})
    out = bundles.build_dense(config["experiment_id"], "all_minilm")
    assert len(list((out / "figures").glob("*.png"))) == 5
    assert len(PdfReader(out / "DENSE.pdf").pages) == 5
    broken = frames["scores"].drop(frames["scores"].query('branch == "control" and step == 270').index[0]).copy()
    with pytest.raises(ValueError):
        export_gallery(config, {**frames, "scores": broken}, specs, "all_minilm", destination)


def test_evaluation_resume_and_checkpoint_isolation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from embedding_geometry import unified_evaluation as e
    config = load_config("configs/unified_v1.json")
    config["evaluation"]["gate_tasks"] = ["TaskOne", "TaskTwo"]
    calls = []
    fail_once = [True]
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(e.mteb, "get_task", lambda name, **kw: SimpleNamespace(
        name=name, metadata=SimpleNamespace(main_score="main_score"), load_data=lambda: None))
    monkeypatch.setattr(e, "task_data_identity", lambda task: {"content": task.name})

    class Evaluator:
        def __init__(self, tasks):
            self.task = tasks[0]

        def run(self, model, output_folder, **kwargs):
            calls.append(self.task.name)
            if self.task.name == "TaskTwo" and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("Simulated interrupted evaluation")
            atomic_json(Path(output_folder) / "result.json", {"task_name": self.task.name,
                "scores": {"test": [{"main_score": .7}]}})

    monkeypatch.setattr(e.mteb, "MTEB", Evaluator)
    identity = {"checkpoint": "a", "step": 68}
    # Prepopulate the independent extra task to avoid downloading AG News.
    atomic_json(tmp_path / "eval/AGNewsClassification.json", {"identity": identity, "task": "AGNewsClassification"})
    with pytest.raises(RuntimeError):
        e.evaluate(model, config, tmp_path / "eval", identity, "gate", tmp_path / "datasets")
    results = e.evaluate(model, config, tmp_path / "eval", identity, "gate", tmp_path / "datasets")
    assert calls == ["TaskOne", "TaskTwo", "TaskTwo"]
    assert len(results) == 3
    with pytest.raises(ValueError):
        e.evaluate(model, config, tmp_path / "eval", {"checkpoint": "b", "step": 68}, "gate", tmp_path / "datasets")


def test_retrieval_dataset_identity_tracks_qrels():
    from types import SimpleNamespace
    from embedding_geometry.unified_evaluation import task_data_identity
    task = SimpleNamespace(corpus={"test": {"d": "doc"}}, queries={"test": {"q": "query"}},
                           relevant_docs={"test": {"q": {"d": 1}}})
    first = task_data_identity(task)
    task.relevant_docs["test"]["q"]["d"] = 0
    assert task_data_identity(task)["relevant_docs"] != first["relevant_docs"]


def test_historical_dense_pixel_equivalence():
    from gallery.bundles import load_bundle
    from gallery.trajectories import compact_trajectories
    import matplotlib.pyplot as plt
    frame = load_bundle()["trajectories"]
    # The test compares the explicit adapter with the unchanged default input.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        first = compact_trajectories(True, data_dir=Path(directory))
        first.canvas.draw()
        a = np.asarray(first.canvas.buffer_rgba()).copy()
        second = compact_trajectories(True, data_dir=Path(directory), frame=frame)
        second.canvas.draw()
        np.testing.assert_array_equal(a, np.asarray(second.canvas.buffer_rgba()))
        plt.close(first)
        plt.close(second)

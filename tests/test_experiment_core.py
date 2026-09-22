from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embedding_geometry.common import normalize_text
from embedding_geometry.geometry import global_geometry, neighborhood_preservation
from embedding_geometry.training import _batches
from embedding_geometry.mixtures import interpolation_residual, mixture_specs, objective_schedule, simplex_weights, step_counts


def test_normalize_text_is_deterministic() -> None:
    assert normalize_text("  Hello,\u00a0WORLD!  ") == "hello world"


def test_classification_batches_are_balanced() -> None:
    rows = [{"text": f"{label}-{index}", "label": label} for label in range(4) for index in range(16)]
    batch = next(_batches(rows, 64, 1, 42, "classification"))
    counts = {label: sum(int(row["label"]) == label for row in batch) for label in range(4)}
    assert counts == {0: 16, 1: 16, 2: 16, 3: 16}


def test_geometry_is_finite_and_self_preserving() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(128, 16)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    metrics, _ = global_geometry(values, k=5, pair_sample_size=1000, seed=42)
    assert all(np.isfinite(value) for value in metrics.values())
    assert neighborhood_preservation(values, values.copy(), 5) == 1.0


def test_simplex_and_experiment_specs() -> None:
    points = simplex_weights(5)
    assert len(points) == 21
    assert len(set(points)) == 21
    assert all(sum(point) == 5 for point in points)
    config = {"mixtures": {"simplex_denominator": 5, "include_barycenter": True, "include_control": True}}
    specs = mixture_specs(config)
    assert len(specs) == 23
    assert len({spec["name"] for spec in specs}) == 23


def test_objective_schedule_has_exact_counts_and_is_deterministic() -> None:
    weights = {"sts": 0.2, "classification": 0.4, "retrieval": 0.4}
    assert step_counts(weights, 270) == {"sts": 54, "classification": 108, "retrieval": 108}
    first = objective_schedule(weights, 270, 42)
    second = objective_schedule(weights, 270, 42)
    third = objective_schedule(weights, 270, 43)
    assert first == second
    assert first != third
    assert {name: first.count(name) for name in weights} == step_counts(weights, 270)


def test_barycenter_schedule_is_balanced() -> None:
    weights = {"sts": 1 / 3, "classification": 1 / 3, "retrieval": 1 / 3}
    assert step_counts(weights, 270) == {"sts": 90, "classification": 90, "retrieval": 90}


def test_interpolation_residual() -> None:
    weights = {"sts": 0.2, "classification": 0.2, "retrieval": 0.6}
    pure = {"sts": 0.1, "classification": -0.2, "retrieval": 0.3}
    expected = 0.2 * 0.1 + 0.2 * -0.2 + 0.6 * 0.3
    assert interpolation_residual(expected + 0.05, weights, pure) == pytest.approx(0.05)

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .common import ROOT, read_jsonl


def _encode(model: SentenceTransformer, texts: list[str], batch_size: int = 256) -> np.ndarray:
    values = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return values.astype(np.float32, copy=False)


def _sample_pairs(size: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, size, size=count)
    right = rng.integers(0, size, size=count)
    mask = left == right
    right[mask] = (right[mask] + 1) % size
    return left, right


def global_geometry(embeddings: np.ndarray, *, k: int, pair_sample_size: int, seed: int) -> dict[str, float]:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)[::-1]
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / total if total > 0 else np.ones_like(eigenvalues) / len(eigenvalues)
    nonzero = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    participation_ratio = float(total**2 / max(float(np.square(eigenvalues).sum()), 1e-12))
    left, right = _sample_pairs(len(embeddings), min(pair_sample_size, max(len(embeddings) * 10, 1000)), seed)
    cosine = np.sum(embeddings[left] * embeddings[right], axis=1)
    squared_distance = np.square(embeddings[left] - embeddings[right]).sum(axis=1)
    uniformity = float(np.log(np.exp(-2.0 * squared_distance).mean() + 1e-12))

    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(embeddings)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    occurrence = np.bincount(indices.ravel(), minlength=len(embeddings)).astype(np.float64)
    mean = occurrence.mean()
    std = occurrence.std()
    hubness_skew = float(np.mean(((occurrence - mean) / std) ** 3)) if std > 0 else 0.0
    return {
        "sample_size": float(len(embeddings)),
        "dimension": float(embeddings.shape[1]),
        "mean_cosine": float(cosine.mean()),
        "cosine_std": float(cosine.std()),
        "uniformity": uniformity,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "top_eigenvalue_share": float(probabilities[0]),
        "top10_eigenvalue_share": float(probabilities[:10].sum()),
        "hubness_skew_k10": hubness_skew,
    }, indices


def neighborhood_preservation(reference: np.ndarray, current: np.ndarray, k: int) -> float:
    size = min(len(reference), len(current))
    reference = reference[:size]
    current = current[:size]
    if size <= k:
        raise ValueError(f"Need more than k={k} aligned samples, got {size}")
    ref_neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(reference).kneighbors(return_distance=False)[:, 1:]
    cur_neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(current).kneighbors(return_distance=False)[:, 1:]
    overlap = [len(set(left) & set(right)) / k for left, right in zip(ref_neighbors, cur_neighbors, strict=True)]
    return float(np.mean(overlap))


def generic_snapshot(
    model: SentenceTransformer,
    config: dict[str, Any],
    *,
    seed: int,
    reference_path: Path | None = None,
    save_reference_path: Path | None = None,
) -> dict[str, float]:
    rows = read_jsonl("data/processed/neutral_diagnostic.jsonl")
    texts = [row["text"] for row in rows[: config["geometry"]["generic_sample_size"]]]
    embeddings = _encode(model, texts, config["evaluation"]["batch_size"])
    metrics, _ = global_geometry(
        embeddings,
        k=config["geometry"]["knn_k"],
        pair_sample_size=config["geometry"]["pair_sample_size"],
        seed=seed,
    )
    if save_reference_path is not None:
        save_reference_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_reference_path, embeddings)
        metrics["neighborhood_preservation"] = 1.0
    elif reference_path is not None and reference_path.exists():
        metrics["neighborhood_preservation"] = neighborhood_preservation(
            np.load(reference_path), embeddings, config["geometry"]["knn_k"]
        )
    return metrics


def task_geometry(model: SentenceTransformer, config: dict[str, Any], seed: int) -> dict[str, float]:
    metrics: dict[str, float] = {}

    sts = read_jsonl("data/processed/sts.jsonl")
    left = _encode(model, [row["sentence1"] for row in sts], config["evaluation"]["batch_size"])
    right = _encode(model, [row["sentence2"] for row in sts], config["evaluation"]["batch_size"])
    scores = np.asarray([row["score"] for row in sts])
    positive = scores >= 0.8
    metrics["sts_alignment"] = float(np.square(left[positive] - right[positive]).sum(axis=1).mean())
    metrics["sts_mean_cosine"] = float(np.sum(left * right, axis=1).mean())

    classification = read_jsonl("data/processed/classification.jsonl")[: config["geometry"]["classification_sample_size"]]
    class_embeddings = _encode(model, [row["text"] for row in classification], config["evaluation"]["batch_size"])
    labels = np.asarray([row["label"] for row in classification])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(labels), size=config["geometry"]["pair_sample_size"])
    j = rng.integers(0, len(labels), size=config["geometry"]["pair_sample_size"])
    distances = 1.0 - np.sum(class_embeddings[i] * class_embeddings[j], axis=1)
    same = labels[i] == labels[j]
    metrics["classification_intra_distance"] = float(distances[same].mean())
    metrics["classification_inter_distance"] = float(distances[~same].mean())
    metrics["classification_margin"] = metrics["classification_inter_distance"] - metrics["classification_intra_distance"]

    retrieval = read_jsonl("data/processed/retrieval.jsonl")
    queries = _encode(model, [row["query"] for row in retrieval], config["evaluation"]["batch_size"])
    positives = _encode(model, [row["positive"] for row in retrieval], config["evaluation"]["batch_size"])
    negatives = _encode(model, [row["negative"] for row in retrieval], config["evaluation"]["batch_size"])
    positive_scores = np.sum(queries * positives, axis=1)
    negative_scores = np.sum(queries * negatives, axis=1)
    metrics["retrieval_positive_cosine"] = float(positive_scores.mean())
    metrics["retrieval_negative_cosine"] = float(negative_scores.mean())
    metrics["retrieval_margin"] = float((positive_scores - negative_scores).mean())
    return metrics


def append_snapshot(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

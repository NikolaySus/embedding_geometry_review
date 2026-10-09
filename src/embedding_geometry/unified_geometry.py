"""Corrected geometry on fixed, identified probes; historical metrics stay intact."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .common import read_jsonl
from .unified_state import digest


VERSION = "geometry_v2_knn_self_excluded"


def neighbors(values, k):
    if len(values) <= k or k < 1:
        raise ValueError("Need 0 < k < number of aligned observations")
    # With X=None sklearn removes self by row identity, including duplicate vectors.
    return NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1).fit(values).kneighbors(return_distance=False)


def preservation(reference, current, k, reference_ids, current_ids):
    if reference.shape != current.shape or list(reference_ids) != list(current_ids):
        raise ValueError("Neighborhood samples must have the same IDs, order and shape")
    left, right = neighbors(reference, k), neighbors(current, k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(left, right, strict=True)]))


def global_metrics(values, k, pair_count, seed):
    centered = values.astype(np.float64) - values.mean(axis=0, dtype=np.float64)
    spectrum = np.linalg.eigvalsh(centered.T @ centered / (len(values) - 1)).clip(0)[::-1]
    total = spectrum.sum()
    if total <= 0:
        raise ValueError("Degenerate covariance: geometric rank is undefined")
    p = spectrum / total
    nz = p[p > 0]
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(values), size=pair_count)
    right = (left + rng.integers(1, len(values), size=pair_count)) % len(values)
    cosine = (values[left] * values[right]).sum(axis=1)
    distances = np.square(values[left] - values[right]).sum(axis=1)
    indices = neighbors(values, k)
    counts = np.bincount(indices.ravel(), minlength=len(values)).astype(float)
    skew = float(np.mean(((counts - counts.mean()) / counts.std()) ** 3)) if counts.std() else 0.0
    return {"effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
            "participation_ratio": float(total ** 2 / np.square(spectrum).sum()),
            "top_eigenvalue_share": float(p[0]), "top10_eigenvalue_share": float(p[:10].sum()),
            "mean_cosine": float(cosine.mean()), "cosine_std": float(cosine.std()),
            "uniformity": float(np.log(np.exp(-2 * distances).mean() + 1e-12)),
            "hubness_skew": skew}, spectrum


def encode(model, texts, batch_size):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def probe_texts(config, smoke=False):
    rows = read_jsonl("data/processed/neutral_diagnostic.jsonl")
    n = min(256, len(rows)) if smoke else config["geometry"]["generic_sample_size"]
    texts = [r["text"] for r in rows[:n]]
    return texts, [digest([i, text]) for i, text in enumerate(texts)]


def snapshot(model, config, reference, reference_ids, smoke=False):
    g, batch = config["geometry"], config["evaluation"]["batch_size"]
    texts, ids = probe_texts(config, smoke)
    values = encode(model, texts, batch)
    pairs = min(2000, g["pair_sample_size"]) if smoke else g["pair_sample_size"]
    metrics, spectrum = global_metrics(values, g["knn_k"], pairs, g["seed"])
    metrics["neighborhood_preservation"] = preservation(reference, values, g["knn_k"], reference_ids, ids)
    limit = 128 if smoke else None
    sts = read_jsonl("data/processed/sts.jsonl")[:limit]
    a = encode(model, [r["sentence1"] for r in sts], batch)
    b = encode(model, [r["sentence2"] for r in sts], batch)
    positive = np.array([r["score"] >= .8 for r in sts])
    if not positive.any():
        raise ValueError("STS probe has no positive pairs")
    metrics["sts_alignment"] = float(np.square(a[positive] - b[positive]).sum(axis=1).mean())
    metrics["sts_mean_cosine"] = float((a * b).sum(axis=1).mean())
    cls = read_jsonl("data/processed/classification.jsonl")[:limit or g["classification_sample_size"]]
    x = encode(model, [r["text"] for r in cls], batch)
    labels = np.array([r["label"] for r in cls])
    rng = np.random.default_rng(g["seed"])
    i = rng.integers(0, len(cls), pairs)
    j = (i + rng.integers(1, len(cls), pairs)) % len(cls)
    same = labels[i] == labels[j]
    if same.all() or not same.any():
        raise ValueError("Classification probe lacks within/between-class pairs")
    dist = 1 - (x[i] * x[j]).sum(axis=1)
    metrics["classification_intra_distance"] = float(dist[same].mean())
    metrics["classification_inter_distance"] = float(dist[~same].mean())
    metrics["classification_margin"] = float(dist[~same].mean() - dist[same].mean())
    retrieval = read_jsonl("data/processed/retrieval.jsonl")[:limit]
    q, p, n = [encode(model, [r[key] for r in retrieval], batch) for key in ("query", "positive", "negative")]
    pos, neg = (q * p).sum(axis=1), (q * n).sum(axis=1)
    metrics.update(retrieval_positive_cosine=float(pos.mean()), retrieval_negative_cosine=float(neg.mean()),
                   retrieval_margin=float((pos - neg).mean()))
    return {"version": VERSION, "probe_ids_hash": digest(ids), "sample_size": len(ids),
            "task_probe_scope": "in_sample_training_data", "metrics": metrics,
            "covariance_eigenvalues": spectrum.tolist()}

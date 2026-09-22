from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mteb
import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ndcg_score

from .common import ROOT, read_jsonl, write_json
from .common import normalize_text


def _encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)


def evaluate_agnews(model: SentenceTransformer, config: dict[str, Any], seed: int) -> dict[str, float]:
    train = read_jsonl("data/processed/classification.jsonl")
    test = list(load_dataset(config["data"]["classification_source"], split="test"))
    batch_size = config["evaluation"]["batch_size"]
    train_embeddings = _encode(model, [row["text"] for row in train], batch_size)
    test_embeddings = _encode(model, [row["text"] for row in test], batch_size)
    classifier = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    classifier.fit(train_embeddings, [int(row["label"]) for row in train])
    predictions = classifier.predict(test_embeddings)
    return {"accuracy": float(accuracy_score([int(row["label"]) for row in test], predictions)), "train_rows": len(train), "test_rows": len(test)}


def evaluate_msmarco_gate(model: SentenceTransformer, config: dict[str, Any], seed: int) -> dict[str, float]:
    rows = read_jsonl("data/processed/retrieval_gate.jsonl")
    batch_size = config["evaluation"]["batch_size"]
    queries = _encode(model, [row["query"] for row in rows], batch_size)
    positives = _encode(model, [row["positive"] for row in rows], batch_size)
    negatives = _encode(model, [row["negative"] for row in rows], batch_size)
    rng = np.random.default_rng(seed)
    reciprocal_ranks = []
    ndcg_values = []
    recall_at_10 = []
    margins = []
    for index, query in enumerate(queries):
        pool = [positives[index], negatives[index]]
        candidates = rng.choice(len(rows), size=98, replace=False)
        pool.extend(negatives[candidates])
        scores = np.asarray(pool) @ query
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        recall_at_10.append(float(rank <= 10))
        relevance = np.zeros((1, len(scores)), dtype=np.float32)
        relevance[0, 0] = 1.0
        ndcg_values.append(float(ndcg_score(relevance, scores[None, :], k=10)))
        margins.append(float(scores[0] - scores[1:].max()))
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg_at_10": float(np.mean(ndcg_values)),
        "recall_at_10": float(np.mean(recall_at_10)),
        "hardest_negative_margin": float(np.mean(margins)),
        "queries": len(rows),
        "candidates_per_query": 100,
    }


def evaluate_clean_sts(model: SentenceTransformer, config: dict[str, Any]) -> dict[str, Any]:
    training_texts = {
        normalize_text(row[key])
        for row in read_jsonl("data/processed/sts.jsonl")
        for key in ("sentence1", "sentence2")
    }
    results = {}
    for task_name in ("STSBenchmark", "SICK-R", "STS12", "STS13", "STS14"):
        task = list(mteb.get_tasks(tasks=[task_name], languages=["eng"]))[0]
        task.load_data()
        rows = [
            row
            for row in task.dataset["test"]
            if normalize_text(row["sentence1"]) not in training_texts
            and normalize_text(row["sentence2"]) not in training_texts
        ]
        left = _encode(model, [row["sentence1"] for row in rows], config["evaluation"]["batch_size"])
        right = _encode(model, [row["sentence2"] for row in rows], config["evaluation"]["batch_size"])
        similarities = np.sum(left * right, axis=1)
        correlation = spearmanr(similarities, [float(row["score"]) for row in rows]).statistic
        results[f"{task_name}.clean_exact"] = {
            "cosine_spearman": float(correlation),
            "rows": len(rows),
            "removed_rows": len(task.dataset["test"]) - len(rows),
        }
    return results


def augment_clean_sts_summaries(config: dict[str, Any]) -> int:
    root = ROOT / "runs" / config["experiment_id"]
    summaries = sorted(root.glob("mteb/full/*/seed_*/*/evaluation_summary.json"))
    completed = 0
    for summary_path in summaries:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if "CleanSTS" in payload:
            completed += 1
            continue
        model_name, seed, branch = payload["model"], int(payload["seed"]), payload["branch"]
        if branch == "m0" and not config["models"][model_name]["build_neutral_m0"]:
            model_path = config["models"][model_name]["model_id"]
        else:
            model_path = root / "full" / model_name / f"seed_{seed}" / branch / "final_model"
        model = SentenceTransformer(str(model_path), device="cuda")
        model.max_seq_length = config["models"][model_name]["max_seq_length"]
        payload["CleanSTS"] = evaluate_clean_sts(model, config)
        write_json(summary_path, payload)
        completed += 1
    return completed


def _extract_mteb_scores(output_folder: Path, tasks: list[str]) -> dict[str, Any]:
    results = {}
    for path in output_folder.rglob("*.json"):
        if path.name == "model_meta.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_name = payload.get("task_name")
        if task_name not in tasks:
            continue
        task = mteb.get_task(task_name)
        main_score = task.metadata.main_score
        split_scores = []
        for entries in payload.get("scores", {}).values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if main_score in entry:
                    split_scores.append(float(entry[main_score]))
        results[task_name] = {
            "main_score": main_score,
            "score": float(np.mean(split_scores)) if split_scores else None,
            "artifact": str(path.relative_to(ROOT)),
        }
    return results


def evaluate_checkpoint(
    config: dict[str, Any],
    model_path: str | Path,
    *,
    model_name: str,
    seed: int,
    branch: str,
    suite: str,
) -> dict[str, Any]:
    model = SentenceTransformer(str(model_path), device="cuda")
    model.max_seq_length = config["models"][model_name]["max_seq_length"]
    all_tasks = config["evaluation"][f"{suite}_tasks"]
    output_folder = ROOT / "runs" / config["experiment_id"] / "mteb" / suite / model_name / f"seed_{seed}" / branch
    reused_summary = None
    tasks = all_tasks
    if suite == "full":
        gate_summary_path = (
            ROOT
            / "runs"
            / config["experiment_id"]
            / "mteb"
            / "gate"
            / model_name
            / f"seed_{seed}"
            / branch
            / "evaluation_summary.json"
        )
        if gate_summary_path.exists():
            reused_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
            gate_tasks = set(config["evaluation"]["gate_tasks"])
            tasks = [task for task in all_tasks if task not in gate_tasks]
    # This is an English-only experiment. Several MTEB task classes are
    # multilingual and otherwise evaluate every available subset by default.
    selected_tasks = list(mteb.get_tasks(tasks=tasks, languages=["eng"]))
    evaluator = mteb.MTEB(tasks=selected_tasks)
    evaluator.run(
        model,
        output_folder=str(output_folder),
        batch_size=config["evaluation"]["batch_size"],
        encode_kwargs={"normalize_embeddings": True},
        verbosity=2,
    )
    mteb_scores = {} if reused_summary is None else dict(reused_summary["mteb"])
    mteb_scores.update(_extract_mteb_scores(output_folder, tasks))
    summary = {
        "model": model_name,
        "seed": seed,
        "branch": branch,
        "suite": suite,
        "mteb_version": mteb.__version__,
        "mteb": mteb_scores,
        "AGNewsClassification": (
            reused_summary["AGNewsClassification"]
            if reused_summary is not None
            else evaluate_agnews(model, config, seed)
        ),
        "MSMARCORerankingGate": (
            reused_summary["MSMARCORerankingGate"]
            if reused_summary is not None
            else evaluate_msmarco_gate(model, config, seed)
        ),
        "reused_gate_results": reused_summary is not None,
    }
    write_json(output_folder / "evaluation_summary.json", summary)
    return summary

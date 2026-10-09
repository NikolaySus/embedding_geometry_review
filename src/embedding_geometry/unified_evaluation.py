"""Checkpoint-scoped evaluation with independently resumable tasks."""

import json
import shutil
import time

import mteb
import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .common import normalize_text, read_jsonl, set_seed
from .evaluation import evaluate_msmarco_gate
from .unified_geometry import encode
from .unified_state import atomic_json, bind, digest, isolated


CLEAN = {"STSBenchmark", "SICK-R", "STS12", "STS13", "STS14"}


def dataset_identity(value):
    if hasattr(value, "_fingerprint"):
        return {"fingerprint": value._fingerprint, "rows": len(value), "features": str(value.features)}
    if isinstance(value, dict):
        return {str(k): dataset_identity(v) for k, v in value.items()}
    raise TypeError(f"Unsupported dataset identity: {type(value)}")


def task_data_identity(task):
    if hasattr(task, "corpus"):
        return {key: digest(getattr(task, key)) for key in ("corpus", "queries", "relevant_docs")}
    return dataset_identity(task.dataset)


def evaluate(model, config, folder, identity, suite, dataset_manifest, smoke=False):
    folder.mkdir(parents=True, exist_ok=True)
    bind(folder / "identity.json", {**identity, "evaluation_seed": config["evaluation"]["seed"],
                                  "mteb": mteb.__version__, "technical_only": smoke})
    batch, seed = config["evaluation"]["batch_size"], config["evaluation"]["seed"]
    names = ["technical_sts", "technical_retrieval", "BIOSSES"] if smoke else config["evaluation"][f"{suite}_tasks"] + ["AGNewsClassification"]
    if suite == "full" and not smoke:
        names += ["MSMARCORerankingGate"]
    if not smoke:
        names += [name + ".clean_exact" for name in config["evaluation"][f"{suite}_tasks"] if name in CLEAN]
    results = []
    for name in names:
        output = folder / f"{name}.json"
        if output.exists():
            payload = json.loads(output.read_text())
            if payload["identity"] != identity:
                raise ValueError(f"Wrong checkpoint for cached evaluation: {output}")
            results.append(payload)
            continue
        start = time.monotonic()
        with isolated(model):
            set_seed(seed)
            if name == "technical_sts":
                rows = read_jsonl("data/processed/sts.jsonl")[:128]
                a, b = [encode(model, [r[k] for r in rows], batch) for k in ("sentence1", "sentence2")]
                score = float(spearmanr((a*b).sum(axis=1), [r["score"] for r in rows]).statistic)
                metric, details = "cosine_spearman", {"rows": len(rows), "scope": "technical_in_sample"}
            elif name == "technical_retrieval":
                rows = read_jsonl("data/processed/retrieval_gate.jsonl")[:128]
                q, p, n = [encode(model, [r[k] for r in rows], batch) for k in ("query", "positive", "negative")]
                score = float(np.mean((q*p).sum(axis=1) > (q*n).sum(axis=1)))
                metric, details = "pair_accuracy", {"rows": len(rows), "scope": "technical_only"}
            elif name == "AGNewsClassification":
                test = load_dataset(config["data"]["classification_source"], split="test")
                bind(dataset_manifest / f"{name}.json", dataset_identity(test))
                train = read_jsonl("data/processed/classification.jsonl")
                a = encode(model, [r["text"] for r in train], batch)
                b = encode(model, list(test["text"]), batch)
                classifier = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
                classifier.fit(a, [r["label"] for r in train])
                score = float(accuracy_score(test["label"], classifier.predict(b)))
                metric, details = "accuracy", {"train_rows": len(train), "test_rows": len(test)}
            elif name == "MSMARCORerankingGate":
                details = evaluate_msmarco_gate(model, config, seed)
                score, metric = details["ndcg_at_10"], "ndcg_at_10"
            else:
                task_name = name.removesuffix(".clean_exact")
                task = mteb.get_task(task_name, languages=["eng"])
                task.load_data()
                ds_identity = task_data_identity(task)
                bind(dataset_manifest / f"{task_name}.json", ds_identity)
                if name.endswith(".clean_exact"):
                    training = {normalize_text(r[k]) for r in read_jsonl("data/processed/sts.jsonl") for k in ("sentence1", "sentence2")}
                    original = task.dataset["test"]
                    rows = [r for r in original if all(normalize_text(r[k]) not in training for k in ("sentence1", "sentence2"))]
                    a, b = [encode(model, [r[k] for r in rows], batch) for k in ("sentence1", "sentence2")]
                    score = float(spearmanr((a*b).sum(axis=1), [r["score"] for r in rows]).statistic)
                    metric = "cosine_spearman"
                    details = {"rows": len(rows), "removed_rows": len(original)-len(rows), "filter": "clean_exact", "dataset": digest(ds_identity)}
                else:
                    task_dir = folder / "mteb" / name
                    # Only the atomic canonical task JSON marks completion. Discard
                    # this checkpoint's unfinished MTEB scratch output before retry.
                    if task_dir.exists():
                        shutil.rmtree(task_dir)
                    mteb.MTEB(tasks=[task]).run(model, output_folder=str(task_dir),
                        batch_size=batch, encode_kwargs={"normalize_embeddings": True},
                        seed=seed, verbosity=0, raise_error=True)
                    metric = task.metadata.main_score
                    matches = []
                    for path in task_dir.rglob("*.json"):
                        raw = json.loads(path.read_text())
                        if raw.get("task_name") == name:
                            values = [entry[metric] for entries in raw["scores"].values()
                                      if isinstance(entries, list) for entry in entries if metric in entry]
                            matches.append(float(np.mean(values)))
                    if len(matches) != 1:
                        raise ValueError(f"Expected one valid MTEB artifact for {name}, got {len(matches)}")
                    score, details = matches[0], {"dataset": digest(ds_identity)}
            if not np.isfinite(score):
                raise ValueError(f"Non-finite evaluation score: {name}")
            payload = {"identity": identity, "task": name, "metric": metric, "score": score,
                       "details": details, "seconds": time.monotonic()-start, "technical_only": smoke}
            atomic_json(output, payload)
            results.append(payload)
    atomic_json(folder / f"complete_{suite}.json", {"identity": identity, "tasks": names})
    return results

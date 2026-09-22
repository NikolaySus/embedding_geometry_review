from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

from .common import ROOT, file_sha256, normalize_text, read_jsonl, write_json, write_jsonl


PROCESSED = ROOT / "data" / "processed"


def _sentences(text: str) -> Iterable[str]:
    text = text.strip()
    if not text or text.startswith("="):
        return
    for value in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text):
        value = re.sub(r"\s+", " ", value).strip()
        if 40 <= len(value) <= 1000 and 6 <= len(value.split()) <= 128:
            yield value


def _reservoir(rows: Iterable[dict[str, Any]], size: int, seed: int, scan_limit: int | None = None):
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen = 0
    keys: set[str] = set()
    for row in rows:
        key = normalize_text(" || ".join(str(value) for value in row.values()))
        if not key or key in keys:
            continue
        keys.add(key)
        seen += 1
        if len(sample) < size:
            sample.append(row)
        else:
            index = rng.randrange(seen)
            if index < size:
                sample[index] = row
        if scan_limit and seen >= scan_limit:
            break
    if len(sample) < size:
        raise RuntimeError(f"Reservoir requested {size} unique rows but found {len(sample)}")
    rng.shuffle(sample)
    return sample, seen


def prepare_neutral(config: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    data_cfg = config["data"]
    total = data_cfg["neutral_m0_examples"] + data_cfg["examples_per_epoch"] + config["geometry"]["generic_sample_size"]
    dataset = load_dataset(
        data_cfg["neutral_source"],
        data_cfg["neutral_config"],
        split="train",
        streaming=True,
    )

    def rows():
        for item in dataset:
            for sentence in _sentences(item["text"]):
                yield {"text": sentence}

    selected, scanned = _reservoir(rows(), total, seed, scan_limit=max(total * 2, 300000))
    m0_end = data_cfg["neutral_m0_examples"]
    control_end = m0_end + data_cfg["examples_per_epoch"]
    outputs = {
        "neutral_m0": selected[:m0_end],
        "neutral_control": selected[m0_end:control_end],
        "neutral_diagnostic": selected[control_end:],
    }
    summary = {"source_rows_scanned": scanned, "parts": {}}
    for name, part in outputs.items():
        path = PROCESSED / f"{name}.jsonl"
        digest = write_jsonl(path, part)
        summary["parts"][name] = {"path": str(path.relative_to(ROOT)), "rows": len(part), "sha256": digest}
    return summary


def prepare_sts(config: dict[str, Any]) -> dict[str, Any]:
    target = config["data"]["examples_per_epoch"]
    rows = list(load_dataset(config["data"]["sts_source"], split="train"))
    clean = []
    seen = set()
    for row in rows:
        key = (normalize_text(row["sentence1"]), normalize_text(row["sentence2"]), float(row["score"]))
        if key in seen:
            continue
        seen.add(key)
        clean.append({"sentence1": row["sentence1"], "sentence2": row["sentence2"], "score": float(row["score"])})
    original = len(clean)
    if original > target:
        clean = clean[:target]
    while len(clean) < target:
        clean.append(dict(clean[len(clean) % original], cycled=True))
    path = PROCESSED / "sts.jsonl"
    digest = write_jsonl(path, clean)
    return {"path": str(path.relative_to(ROOT)), "rows": len(clean), "unique_source_rows": original, "cycled_rows": target - original, "sha256": digest}


def prepare_classification(config: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    target = config["data"]["examples_per_epoch"]
    per_class = target // 4
    dataset = load_dataset(config["data"]["classification_source"], split="train", streaming=True)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)
    for row in dataset:
        label = int(row["label"])
        key = normalize_text(row["text"])
        if key and key not in seen[label] and len(grouped[label]) < per_class:
            seen[label].add(key)
            grouped[label].append({"text": row["text"], "label": label})
        if len(grouped) == 4 and all(len(values) == per_class for values in grouped.values()):
            break
    rows = [row for label in sorted(grouped) for row in grouped[label]]
    random.Random(seed).shuffle(rows)
    if len(rows) != target:
        raise RuntimeError(f"Expected {target} balanced AG News rows, found {len(rows)}")
    path = PROCESSED / "classification.jsonl"
    digest = write_jsonl(path, rows)
    return {"path": str(path.relative_to(ROOT)), "rows": len(rows), "class_counts": dict(Counter(row["label"] for row in rows)), "sha256": digest}


def prepare_retrieval(config: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    data_cfg = config["data"]
    dataset = load_dataset(
        data_cfg["retrieval_source"],
        data_cfg["retrieval_config"],
        split="train",
        streaming=True,
    )
    train_size = data_cfg["examples_per_epoch"]
    gate_size = 1000
    selected, scanned = _reservoir(
        ({"query": row["query"], "positive": row["positive"], "negative": row["negative"]} for row in dataset),
        train_size + gate_size,
        seed,
        data_cfg["retrieval_scan_limit"],
    )
    path = PROCESSED / "retrieval.jsonl"
    digest = write_jsonl(path, selected[:train_size])
    gate_path = PROCESSED / "retrieval_gate.jsonl"
    gate_digest = write_jsonl(gate_path, selected[train_size:])
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": train_size,
        "source_rows_scanned": scanned,
        "sha256": digest,
        "gate_path": str(gate_path.relative_to(ROOT)),
        "gate_rows": gate_size,
        "gate_sha256": gate_digest,
    }


def prepare_all(config: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    summary = {
        "neutral": prepare_neutral(config, seed),
        "sts": prepare_sts(config),
        "classification": prepare_classification(config, seed),
        "retrieval": prepare_retrieval(config, seed),
    }
    write_json(PROCESSED / "manifest.json", summary)
    return summary


def _shingles(text: str, width: int = 5) -> set[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _minhash(text: str, num_perm: int = 64) -> MinHash:
    value = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        value.update(shingle.encode("utf-8"))
    return value


def audit_processed(config: dict[str, Any]) -> dict[str, Any]:
    import json

    sources = {}
    for name in ("neutral_m0", "neutral_control", "neutral_diagnostic", "sts", "classification", "retrieval"):
        path = PROCESSED / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        # Stream JSONL records: the neutral corpus is deliberately large and
        # materialising the complete file as one string needlessly doubles RAM.
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        texts = []
        for row in rows:
            texts.extend(str(value) for key, value in row.items() if key not in {"label", "score", "cycled"})
        normalized = [normalize_text(value) for value in texts if normalize_text(value)]
        sources[name] = {
            "rows": len(rows),
            "texts": len(normalized),
            "unique_texts": len(set(normalized)),
            "duplicate_texts": len(normalized) - len(set(normalized)),
            "sha256": file_sha256(path),
            "normalized_set": set(normalized),
        }

    overlaps = {}
    names = list(sources)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            count = len(sources[left]["normalized_set"] & sources[right]["normalized_set"])
            if count:
                overlaps[f"{left}::{right}"] = count

    # Near-duplicate audit is restricted to specialization datasets to keep it bounded.
    lsh = MinHashLSH(threshold=0.9, num_perm=64)
    near_duplicates = []
    indexed: dict[str, str] = {}
    for source_name in ("sts", "classification", "retrieval"):
        for index, text in enumerate(sorted(sources[source_name]["normalized_set"])):
            if len(text.split()) < 5:
                continue
            key = f"{source_name}:{index}"
            signature = _minhash(text)
            matches = [item for item in lsh.query(signature) if not item.startswith(f"{source_name}:")]
            if matches and len(near_duplicates) < 200:
                near_duplicates.append({"source": key, "matches": matches[:5], "text": text[:300]})
            lsh.insert(key, signature)
            indexed[key] = text

    report = {
        "sources": {name: {key: value for key, value in data.items() if key != "normalized_set"} for name, data in sources.items()},
        "exact_cross_source_overlaps": overlaps,
        "near_duplicate_candidates": near_duplicates,
        "near_duplicate_threshold": 0.9,
        "decision": "pass" if not overlaps else "review",
    }
    write_json(ROOT / "reports" / "pilot_v1_data_audit.json", report)
    return report


def audit_evaluation_overlap(config: dict[str, Any]) -> dict[str, Any]:
    """Audit exact normalized text overlap between training and evaluation data."""
    import mteb

    skip_keys = {"label", "labels", "score", "scores", "relevant_docs", "id", "_id"}

    def text_values(value, key: str | None = None):
        if key in skip_keys:
            return
        if isinstance(value, str):
            normalized = normalize_text(value)
            if normalized:
                yield normalized
        elif isinstance(value, dict):
            for child_key, child in value.items():
                yield from text_values(child, str(child_key).lower())
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from text_values(child, key)

    training: dict[str, set[str]] = {}
    for source_name in ("neutral_m0", "neutral_control", "sts", "classification", "retrieval"):
        values: set[str] = set()
        for row in read_jsonl(f"data/processed/{source_name}.jsonl"):
            values.update(text_values(row))
        training[source_name] = values

    def compare(task_name: str, values) -> dict[str, Any]:
        counts = {name: 0 for name in training}
        examples: dict[str, list[str]] = {name: [] for name in training}
        seen: set[str] = set()
        total = 0
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            total += 1
            for source_name, source_values in training.items():
                if value in source_values:
                    counts[source_name] += 1
                    if len(examples[source_name]) < 5:
                        examples[source_name].append(value[:300])
        return {
            "task": task_name,
            "unique_evaluation_texts": total,
            "exact_overlap_counts": {key: value for key, value in counts.items() if value},
            "examples": {key: value for key, value in examples.items() if value},
        }

    task_reports = []
    tasks = list(mteb.get_tasks(tasks=config["evaluation"]["full_tasks"], languages=["eng"]))
    for task in tasks:
        task.load_data()
        values = []
        if task.dataset is not None:
            for split in task.metadata.eval_splits:
                if split in task.dataset:
                    for row in task.dataset[split]:
                        values.extend(text_values(row))
        else:
            for attribute in ("corpus", "queries"):
                collection = getattr(task, attribute, {})
                for split in task.metadata.eval_splits:
                    if split in collection:
                        values.extend(text_values(collection[split]))
        task_reports.append(compare(task.metadata.name, values))

    ag_test = load_dataset(config["data"]["classification_source"], split="test")
    task_reports.append(compare("AGNewsClassification.custom_test", (text for row in ag_test for text in text_values(row))))
    retrieval_gate = read_jsonl("data/processed/retrieval_gate.jsonl")
    task_reports.append(compare("MSMARCORerankingGate.custom_test", (text for row in retrieval_gate for text in text_values(row))))

    overlaps = [report for report in task_reports if report["exact_overlap_counts"]]
    report = {
        "scope": "exact normalized text overlap; English evaluation subsets only",
        "tasks_checked": len(task_reports),
        "training_sources": {name: len(values) for name, values in training.items()},
        "tasks_with_overlap": len(overlaps),
        "results": task_reports,
        "decision": "pass" if not overlaps else "review",
        "limitations": "Near-duplicate and semantic overlap are not covered by this exact-match audit.",
    }
    write_json(ROOT / "reports" / "pilot_v1_evaluation_overlap_audit.json", report)
    return report

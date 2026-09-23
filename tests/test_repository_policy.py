from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")
    return tmp_path


@pytest.mark.parametrize("path", [
    ".venv/bin/python", "nested/.venv/pyvenv.cfg", "src/__pycache__/module.pyc",
    "build/gallery/interactive/index.html", "build/full/data/table.csv",
    "build/compact/article.docx", "nested/figure.png", "nested/figure.pdf",
    "nested/article.docx", "nested/figure.svg", ".env",
    "runs/example/seed_42/sts/final_model/config.json",
    "runs/example/seed_42/sts/final_model/model.safetensors",
    "runs/example/seed_42/sts/checkpoints/step.pt",
    "runs/example/seed_42/m0_reference_embeddings.npy",
    "data/processed/sts.jsonl", "data/raw/source.csv", "cache/hf/file.json",
])
def test_generated_or_large_artifacts_ignored(repository, path):
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=repository)
    assert result.returncode == 0, path


@pytest.mark.parametrize("path", [
    "uv.lock", "pyproject.toml", ".python-version", "AGENTS.md", ".env.example",
    "src/embedding_geometry/training.py", "src/build_article.py", "src/build_gallery.py",
    "manuscript/full.md", "manuscript/compact.md", "references/references.json",
    "assets/gallery/cameras/selected.json", "assets/compact/data/geometry.csv",
    "reports/pilot_v1_data_audit.json", "configs/pilot_v1.json",
    "runs/example/seed_42/sts/train_metrics.jsonl", "runs/example/train.log",
    "runs/example/seed_42/sts/geometry_trajectory.csv",
    "runs/example/seed_42/sts/manifest.json",
    "runs/example/mteb/task.json", "data/processed/manifest.json",
    "docs/migration_manifest.json",
])
def test_research_and_sources_not_ignored(repository, path):
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=repository)
    assert result.returncode == 1, path


def test_migration_storage_matches_git_policy(repository):
    manifest = json.loads((ROOT / "docs/migration_manifest.json").read_text())
    paths = [entry["path"] for entry in manifest["files"]]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"], cwd=repository,
        input=("\0".join(paths) + "\0").encode("utf-8"), capture_output=True,
    )
    assert result.returncode in (0, 1), result.stderr
    ignored = set(result.stdout.decode("utf-8").split("\0"))
    for entry in manifest["files"]:
        assert (entry["path"] in ignored) == (entry["storage"] == "local"), entry["path"]

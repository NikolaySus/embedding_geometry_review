"""Record or verify byte-preserved research artifacts, without rewriting them."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "migration_manifest.json"
UNCHANGED = (
    "src/embedding_geometry", "src/experiment.py", "configs", "runs", "reports",
    "references", "data", "tests/test_experiment_core.py", "pyproject.toml",
    "uv.lock", ".python-version",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_only(path: Path) -> bool:
    return (
        "final_model" in path.parts
        or path.suffix in {".npy", ".npz", ".pt", ".safetensors", ".arrow", ".parquet"}
        or (path.parts[:2] in {("data", "raw"), ("data", "processed")}
            and path.name != "manifest.json")
    )


def files(root: Path):
    for relative in UNCHANGED:
        start = root / relative
        candidates = start.rglob("*") if start.is_dir() else [start]
        for path in sorted(candidates):
            if path.is_file() and "__pycache__" not in path.parts:
                yield path.relative_to(root)


def record(source: Path) -> None:
    entries = []
    for relative in files(source):
        original, migrated = source / relative, ROOT / relative
        digest = sha256(original)
        if not migrated.is_file() or sha256(migrated) != digest:
            raise SystemExit(f"Not copied exactly: {relative}")
        entries.append({
            "path": relative.as_posix(), "bytes": original.stat().st_size,
            "sha256": digest, "storage": "local" if local_only(relative) else "git",
        })
    result = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "scope": "Byte-preserved research; publication source transformations documented separately",
        "baselines": {"full": "experimental_v4", "compact": "experimental_v5", "gallery": "refined_v3"},
        "files": entries,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"recorded": len(entries), "storage": dict(Counter(e["storage"] for e in entries))}))


def verify(include_local: bool, source: Path | None) -> None:
    manifest = json.loads(MANIFEST.read_text())
    checked, skipped, errors = 0, 0, []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if entry["storage"] == "local" and not include_local:
            skipped += 1
            continue
        if not path.is_file() or sha256(path) != entry["sha256"]:
            errors.append(f"Target missing or changed: {entry['path']}")
        if source is not None:
            original = source / entry["path"]
            if not original.is_file() or sha256(original) != entry["sha256"]:
                errors.append(f"Source missing or changed: {entry['path']}")
        checked += 1
    print(json.dumps({"checked": checked, "local_skipped": skipped, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Record the initial migration; not for routine use")
    parser.add_argument("--source", type=Path, help="Optionally compare to the untouched source repository")
    parser.add_argument("--include-local", action="store_true", help="Also verify ignored weights, corpora and arrays")
    args = parser.parse_args()
    if args.record:
        if args.source is None:
            parser.error("--record requires --source")
        if MANIFEST.exists():
            parser.error("Migration manifest already exists; do not overwrite historical evidence")
        record(args.source)
    else:
        verify(args.include_local, args.source)


if __name__ == "__main__":
    main()

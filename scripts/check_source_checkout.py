"""Build publications using only files visible to Git, without a commit or weights."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "source-checkout-check"


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    paths = sorted(set(p for p in result.stdout.decode().split("\0") if p))
    largest = sorted(
        ((ROOT / p).stat().st_size, p) for p in paths if (ROOT / p).is_file()
    )[-10:]
    excessive = [(size, path) for size, path in largest if size > 10 * 1024**2]
    if excessive:
        raise SystemExit(f"Review unusually large Git candidates before copying: {excessive}")
    checkout = OUT / "checkout"
    if checkout.exists():
        shutil.rmtree(checkout)
    checkout.mkdir(parents=True)
    for relative in paths:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f"Unsupported source entry: {relative}")
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    commands = [
        ["scripts/verify_migration.py"],
        ["src/build_article.py", "--variant", "full"],
        ["src/build_article.py", "--variant", "compact"],
        ["src/build_gallery.py"],
        ["src/validate_publication.py", "--all"],
    ]
    completed = []
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        log = OUT / f"step-{index}.log"
        with log.open("w") as stream:
            process = subprocess.run(
                [sys.executable, *command], cwd=checkout, env=env,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        completed.append({"command": command, "exit_code": process.returncode, "log": log.name})
        if process.returncode:
            print(log.read_text()[-6000:])
            raise SystemExit(process.returncode)
    summary = {
        "status": "passed", "source_files": len(paths),
        "source_bytes": sum((ROOT / p).stat().st_size for p in paths),
        "largest_git_candidates": [{"bytes": size, "path": path} for size, path in reversed(largest)],
        "checks": completed,
        "conditions": "Fresh Git-visible files only; no weights, corpora or prior build outputs; offline HF; GPU hidden",
    }
    (OUT / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

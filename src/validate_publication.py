"""Validate article outputs and optionally the independently built gallery."""

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--variant", choices=("full", "compact"))
    choice.add_argument("--all", action="store_true")
    args = parser.parse_args()
    for variant in ("full", "compact") if args.all else (args.variant,):
        if variant == "full":
            from publication.validate_full import main as validate
        else:
            from publication.validate_compact import main as validate
        validate()
    if args.all:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("validate_gallery.py"))],
            check=True,
        )


if __name__ == "__main__":
    main()

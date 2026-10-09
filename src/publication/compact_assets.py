"""Standalone compact figure generation from the pinned unified experiment."""
from .unified_assets import ROOT, COMPACT_IDS as FIGURE_IDS, main as build
OUT = ROOT / "build/compact"

def main():
    return build("compact", OUT)

if __name__ == "__main__":
    main()

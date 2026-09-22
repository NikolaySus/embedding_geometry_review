"""Build a publication variant from source data and manuscript."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from publication import document


ROOT = Path(__file__).resolve().parents[1]


def build(variant: str) -> Path:
    if variant not in {"full", "compact"}:
        raise ValueError(f"Unknown article variant: {variant}")
    # Restore plotting defaults after each generator, also for in-process builds.
    if variant == "full":
        from publication import full_assets, mixture_assets

        with matplotlib.rc_context():
            full_assets.main()
        with matplotlib.rc_context():
            mixture_assets.main()
    else:
        from publication import compact_assets

        with matplotlib.rc_context():
            compact_assets.main()
    output = ROOT / "build" / variant
    docx = document.build_docx(
        ROOT / "manuscript" / f"{variant}.md",
        ROOT / "references" / "references.json",
        output / f"cross_task_interference_{variant}.docx",
    )
    pdf = document.export_pdf(docx)
    pages = document.render_pages(pdf, output / "pages")
    sheets = document.build_contact_sheets(pages, output / "contact_sheets")
    print(f"DOCX: {docx}\nPDF: {pdf}\nPages: {len(pages)}\nContact sheets: {len(sheets)}")
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("full", "compact"))
    args = parser.parse_args()
    build(args.variant)


if __name__ == "__main__":
    main()

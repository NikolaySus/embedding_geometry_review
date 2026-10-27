"""Build a publication variant from source data and manuscript."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from publication import document


ROOT = Path(__file__).resolve().parents[1]


def build(variant: str) -> Path:
    if variant not in {"full", "compact", "ultracompact"}:
        raise ValueError(f"Unknown article variant: {variant}")
    # Restore plotting defaults after each generator, also for in-process builds.
    from publication import unified_assets
    with matplotlib.rc_context():
        if variant == "ultracompact":
            from publication import ultracompact
            values = ultracompact.main()
        else:
            values = unified_assets.main(variant)
    output = ROOT / "build" / variant
    source = ROOT / "manuscript" / f"{variant}.md"
    text = source.read_text(encoding='utf-8')
    if variant == "ultracompact":
        from publication import conference
        text = conference.prepare_text(text)
    text = unified_assets.render_text(text, values)
    # The resolved manuscript is reproducible output, not a second editable source.
    text = text.replace(f"../build/{variant}/figures/", "figures/")
    resolved = output / "manuscript.md"
    resolved.write_text(text, encoding="utf-8")
    if variant == "ultracompact":
        docx = conference.build_docx(resolved, output / f"cross_task_interference_{variant}.docx")
    else:
        docx = document.build_docx(
            resolved,
            ROOT / "references" / "references.json",
            output / f"cross_task_interference_{variant}.docx",
            references_new_page=False,
        )
    pdf = document.export_pdf(docx)
    pages = document.render_pages(pdf, output / "pages")
    sheets = document.build_contact_sheets(pages, output / "contact_sheets")
    print(f"DOCX: {docx}\nPDF: {pdf}\nPages: {len(pages)}\nContact sheets: {len(sheets)}")
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("full", "compact", "ultracompact"))
    args = parser.parse_args()
    build(args.variant)


if __name__ == "__main__":
    main()

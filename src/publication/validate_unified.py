"""Source, evidence, and document consistency checks for both variants."""

import hashlib
import json
import re
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from pypdf import PdfReader

from .unified_assets import ROOT, SETTINGS, COMPACT_IDS, FULL_IDS, correlations, render_text
from .unified_assets import evidence as calculate_evidence
from gallery.bundles import load_bundle


def validate(variant):
    output = ROOT / "build" / variant
    data = output / "data"
    evidence = json.loads((data / "evidence.json").read_text())
    with TemporaryDirectory() as temporary:
        fresh = calculate_evidence(load_bundle(SETTINGS["experiment"], SETTINGS["model"]), Path(temporary))
    assert evidence == fresh, "Numeric assertions do not match current source data"
    expected = render_text((ROOT / "manuscript" / f"{variant}.md").read_text(), evidence)
    expected = expected.replace(f"../build/{variant}/figures/", "figures/")
    text = (output / "manuscript.md").read_text()
    assert expected == text, "Stale rendered manuscript"
    assert not re.search(r"Raw MiniLM|raw_minilm|нескольких компактных моделей", text)
    assert re.findall(r"\{\{(.*?)\}\}", text) == ["REFERENCES"]
    provenance = json.loads((data / "provenance.json").read_text())
    for key, value in SETTINGS.items():
        assert provenance[key] == value
    for path, digest in provenance["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    delta = pd.read_csv(data / "deltas.csv")
    assert len(delta) == 69 and set(delta.seed) == {42, 43, 44}
    absolute = pd.read_csv(data / "absolute.csv")
    assert len(absolute) == 72
    assert (absolute[absolute.branch == "m0"].drop(columns=["branch", "seed"]).std() < 1e-12).all()
    corr = correlations(delta)
    pd.testing.assert_frame_equal(corr, pd.read_csv(data / "correlations.csv"))
    assert list(corr.n) == [12, 12, 9, 9]
    trajectory = pd.read_csv(data / "quality_trajectories.csv")
    assert len(trajectory) == 360 and set(trajectory.suite) == {"gate"}
    assert set(trajectory.step) == {0, 68, 135, 270}
    tasks = pd.read_csv(data / "selected_task_scores.csv")
    assert set(tasks[tasks.branch != "m0"].step) == {270}
    assert len(tasks) == 23 * (6 * 3 + 1)
    ids = COMPACT_IDS if variant == "compact" else FULL_IDS
    manifest = json.loads((data / "figure_manifest.json").read_text())
    assert [r["id"] for r in manifest] == list(ids)
    assert re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) == [f"figures/{i}.png" for i in ids]
    assert re.findall(r"\*Рисунок (\d+) —", text) == [str(i) for i in range(1, len(ids) + 1)]
    for row in manifest:
        page = PdfReader(output / "figures" / f"{row['id']}.pdf").pages[0]
        assert np.isclose(float(page.mediabox.width), 72 * row["width_inches"])
        assert np.isclose(float(page.mediabox.height), 72 * row["height_inches"])
    doc = Document(output / f"cross_task_interference_{variant}.docx")
    pdf = PdfReader(output / f"cross_task_interference_{variant}.pdf")
    assert len(doc.inline_shapes) == len(ids)
    assert len(list((output / "pages").glob("page-*.png"))) == len(pdf.pages)
    assert all(np.isclose(shape.width / 914400, 6.2) for shape in doc.inline_shapes)
    pdf_text = "\n".join(page.extract_text() for page in pdf.pages)
    assert "{{" not in pdf_text and "editorial:" not in pdf_text
    for page in pdf.pages:
        if re.search(r"Рисунок\s+\d+\s+[—-]", page.extract_text()):
            assert len(page.images), "Detached figure caption"
    words = len(re.findall(r"\b[\w-]+\b", text))
    print(f"{variant}: {words} words, {len(pdf.pages)} pages, {len(ids)} figures, {len(doc.tables)} tables")

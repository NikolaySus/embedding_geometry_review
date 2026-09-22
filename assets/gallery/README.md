# Gallery Sources

Run from the project root:

```sh
uv run python src/build_gallery.py
uv run python src/validate_gallery.py
uv run pytest tests/test_gallery.py
```

The three CSVs in `data/` are byte-exact compact experimental inputs.
`inputs.json` records their SHA-256 digests. No derived CSVs, figure JSON,
HTML, images, or PDFs are required as inputs. Means, SDs, residuals, and
Delaunay connectivity are regenerated from these CSVs.

`layouts.json` records the selected surface layout settings (including the
original overlap diagnostics). `examples/font8-az60.camera.json` is an
8-point, 60-degree export sample from a previous browser check. It is NOT
known to be author-selected and is never applied or exported automatically.
Default cameras are generated under `build/gallery/interactive/`.

All generated outputs live in `build/gallery/`: `COMPARISON.pdf` (19 pages),
`manifest.json`, `figures/`, derived `data/`, and `interactive/index.html`.
The interactive HTML files embed Plotly and open directly without a server.
The refined v3 comparison is the migration baseline. Its variant IDs and
controls are preserved; cover/index titles are neutral. There are no
versioned packages or output directories.

To render a camera downloaded from a panel:

```sh
uv run python src/build_gallery.py --camera path/to/panel.camera.json
```

Camera export also constructs its figure directly from source inputs; it
does not require a previous build. Static camera export needs a Chrome or
Chromium installation discoverable by Kaleido (`BROWSER_PATH` is supported).

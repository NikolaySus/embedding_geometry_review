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

All generated outputs live in `build/gallery/`: `COMPARISON.pdf` (10 pages, 9 variants),
`manifest.json`, `figures/`, derived `data/`, and `interactive/index.html`.
The interactive HTML files embed Plotly and open directly without a server.
The refined v3 comparison is the migration baseline. Figure 1 now has only the selected absolute table; other variant IDs and
controls are preserved; cover/index titles are neutral. There are no
versioned packages or output directories.

## Selected absolute-value table

`absolute_tables.json` selects `1C-09-absolute-unit` (quality 0–1) as the sole
figure 1 variant, based on the numeric-table design of `1C-01-symmetric-uniform`.
Retired table PDFs, PNGs and scale metadata are removed during gallery builds.
Color bars sit close to the table; cell dimensions and text sizes are preserved
while the figure height is reduced from 8.0 to 7.6 inches. All five quality
columns share one RdBu scale. The four geometry columns use distinct sequential
palettes and explicitly labelled independent ranges. No displayed mean is clipped.
These are absolute values for all nine columns, with an added M0 reference row.
For absolute quality, red/blue means lower/higher, not degradation/improvement;
equal numbers across different metrics do not imply equal utility. The 0–1 scale
is a display range containing these observations, not the theoretical range of Spearman.

Inputs are the preserved `reports/pilot_v2_mixtures/score_deltas.csv` (`score`,
not `delta_vs_m0`) and `geometry.csv`. Task membership is explicit in the JSON:
clean STS subsets, classification transfer without AG News, and clean retrieval
without the MS MARCO gate. Scores are averaged over tasks within each seed before
computing means and sample SDs over the three seeds. Every reconstructed per-seed
delta is checked against the original gallery CSV. kNN remains the overlap with
M0 by definition; M0 SD reflects repeated evaluation, not independent training.

Generated absolute data, summaries and scale metadata are under `build/gallery/data/`.
The remaining surface/trajectory figures and interactive controls retain their semantics.

To render a camera downloaded from a panel:

```sh
uv run python src/build_gallery.py --camera path/to/panel.camera.json
```

Camera export also constructs its figure directly from source inputs; it
does not require a previous build. Static camera export needs a Chrome or
Chromium installation discoverable by Kaleido (`BROWSER_PATH` is supported).

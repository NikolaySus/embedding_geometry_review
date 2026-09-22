# Rules for this research repository

## Research integrity

- Use `uv` and the committed `uv.lock`. Do not upgrade scientific dependencies
  as part of unrelated edits.
- Training/evaluation code is in `src/embedding_geometry/`, with the CLI at
  `src/experiment.py`. Existing experiment IDs are immutable identifiers, not
  document revisions. Preserve their configuration, raw metrics and audit flags.
- Do not run training, long evaluations, or delete checkpoints just to rebuild
  documents. Existing tables and figures must come from recorded observations.
- Historical reports and run manifests may contain old source paths. Preserve
  their bytes; use `docs/MIGRATION.md` to interpret provenance.

## Publication versions

- There are two explicitly requested article variants: `manuscript/full.md`
  (initially experimental v4) and `manuscript/compact.md` (initially v5).
- The single current gallery starts from refined v3. Its alternatives are figure
  design choices, not new document revisions.
- Git stores revision history. Update existing files rather than creating v6,
  v7, backup copies, or parallel manuscripts unless the user explicitly asks.
- Preserve current figure semantics and controls when changing the build system.

## Storage and delivery

- Keep article-related files inside this repository. Never read the old repository
  during ordinary builds. An explicit migration verification is the exception.
- Source Markdown, code, numeric results, bibliography and curated figure/camera
  settings belong in Git. Put generated documents, images, interactive HTML and
  render checks under `build/`, which is ignored.
- Keep model weights, corpora, embedding caches and `.venv` local and ignored.
  Do not ignore `runs/` wholesale: its CSV/JSON/JSONL metrics must remain visible.
- Before preparing a commit, run the ignore-policy tests and inspect file sizes.
- Do not commit or push without a user request.

## Checks

```bash
uv sync --locked
uv run pytest
uv run python scripts/verify_migration.py
uv run python src/build_article.py --variant full
uv run python src/build_article.py --variant compact
uv run python src/build_gallery.py
uv run python src/validate_publication.py --all
```

Builds require LibreOffice, Poppler and a Chrome-compatible browser for Kaleido.
Inspect rendered pages as well as numerical validation results.

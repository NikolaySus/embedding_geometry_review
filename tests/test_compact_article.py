"""The compact publication reuses gallery rendering without gallery output files."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops

from gallery import tables, trajectories
from gallery.data import absolute_config, load_absolute_data, load_data, G, Q
from gallery.surfaces import compact_surface
from publication import compact_assets


def test_compact_build_without_gallery_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(compact_assets, 'OUT', tmp_path / 'compact')
    forbidden = tmp_path / 'no-gallery'
    monkeypatch.setattr(tables, 'OUT', forbidden)
    monkeypatch.setattr(trajectories, 'OUT', forbidden)
    compact_assets.main()
    assert not forbidden.exists()
    output = tmp_path / 'compact'
    assert {p.stem for p in (output/'figures').glob('*.png')} == set(compact_assets.FIGURE_IDS)
    assert not list(output.rglob('*.html'))
    manifest = json.loads((output/'data/figure_manifest.json').read_text())
    assert [r['id'] for r in manifest] == list(compact_assets.FIGURE_IDS)
    np.testing.assert_allclose([r['height_inches'] for r in manifest], [7.1, 6.65, 6.65, 5.2, 6.65])
    _, mean, residual, triangles = load_data()
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.labelsize':8})
        variants = absolute_config()['variants']
        figures = [
            tables.absolute_table(load_absolute_data(), next(v for v in variants if v['id']==compact_assets.FIGURE_IDS[0]), data_dir=output/'data'),
            compact_surface(mean[G], triangles, '2D', dense=True)[0],
            compact_surface(mean[Q[1:]], triangles, '3E', dense=True)[0],
            trajectories.compact_trajectories(dense=True, data_dir=output/'data'),
            compact_surface(residual, triangles, '5A', dense=True)[0],
        ]
        for identity, fig in zip(compact_assets.FIGURE_IDS, figures):
            reference = tmp_path / f'{identity}.png'
            fig.savefig(reference, dpi=180)
            plt.close(fig)
            with Image.open(reference) as a, Image.open(output/'figures'/f'{identity}.png') as b:
                assert a.size == b.size
                assert ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is None


def test_retained_evidence_matches_previous_numeric_sources():
    details, stats = compact_assets.evidence()
    old = pd.read_csv(compact_assets.SOURCE/'correlations.csv')
    np.testing.assert_allclose([r['r'] for r in details['correlations']], old.r, rtol=1e-12)
    assert all(r['n'] == 12 for r in details['correlations'])
    pd.testing.assert_frame_equal(stats, pd.read_csv(compact_assets.SOURCE/'pareto_summary.csv', header=[0,1], index_col=0), check_names=False)
    pd.testing.assert_frame_equal(compact_assets.interaction_summary(), pd.read_csv(compact_assets.SOURCE/'interaction_examples.csv', header=[0,1], index_col=0), check_names=False)
    text = (compact_assets.ROOT/'manuscript/compact.md').read_text()
    assert all(row in text for row in compact_assets.interaction_markdown_rows())

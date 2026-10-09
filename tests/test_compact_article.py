"""Publication sources share one experiment and preserve gate semantics."""
import json
import pandas as pd
import pytest
from publication import compact_assets
from publication.unified_assets import MODEL, correlations, render_text, SETTINGS

def test_compact_build_without_gallery_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(compact_assets, "OUT", tmp_path / "compact")
    compact_assets.main()
    output = tmp_path / "compact"
    assert {p.stem for p in (output / "figures").glob("*.png")} == set(compact_assets.FIGURE_IDS)
    assert not list(output.rglob("*.html"))
    assert json.loads((output / "data/provenance.json").read_text())["protocol_hash"] == SETTINGS["protocol_hash"]
    assert len(pd.read_csv(output / "data/quality_trajectories.csv")) == 360

def test_correlation_sensitivity_uses_only_endpoints():
    result = correlations(pd.read_csv(MODEL / "deltas.csv"))
    assert list(result.n) == [12, 12, 9, 9]
    assert result.iloc[0].r > .9
    assert abs(result.iloc[2].r) < .1
    assert result.iloc[3].r > .9

def test_unknown_evidence_token_is_an_error():
    assert render_text("{{X}} {{REFERENCES}}", {"X": "value"}) == "value {{REFERENCES}}"
    with pytest.raises(KeyError):
        render_text("{{MISSING}}", {})

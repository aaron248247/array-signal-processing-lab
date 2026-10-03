"""Integrity checks for stored result files (guards against corruption, not a scientific assertion)."""

import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"
SCAN = RESULTS / "cfar_masking_seed_scan.json"


@pytest.mark.skipif(not SCAN.exists(), reason="masking seed scan not generated")
def test_cfar_masking_seed_scan_file_consistent():
    rec = json.loads(SCAN.read_text())
    seeds = rec["config"]["seeds"]
    assert seeds == list(range(20))
    m = rec["metrics"]
    assert m["n_realisations"] == len(seeds)
    selected = [g for g in m["geometries"] if g["selected_for_figure_10"]]
    assert len(selected) == 1
    with np.load(RESULTS / "cfar_masking_seed_scan.npz") as z:
        np.testing.assert_array_equal(z["seeds"], seeds)
        for g in m["geometries"]:
            rows = g["per_seed"]
            assert [r["seed"] for r in rows] == seeds  # exactly one result per seed, in order
            assert g["n_realisations"] == len(rows)
            assert g["ca_detections"] == sum(r["ca_detected"] for r in rows)
            assert g["os_detections"] == sum(r["os_detected"] for r in rows)
            assert g["ca_detection_fraction"] == pytest.approx(g["ca_detections"] / len(rows))
            assert g["os_detection_fraction"] == pytest.approx(g["os_detections"] / len(rows))
            tag = f"off{g['masker_offset_cells']}_m{int(abs(g['masker_snr_db']))}"
            np.testing.assert_array_equal(z[f"ca_detected_{tag}"], [r["ca_detected"] for r in rows])
            np.testing.assert_array_equal(z[f"os_detected_{tag}"], [r["os_detected"] for r in rows])
    assert m["selected_ca_detections"] == selected[0]["ca_detections"]
    assert m["selected_os_detections"] == selected[0]["os_detections"]

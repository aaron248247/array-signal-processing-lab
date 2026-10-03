"""Seed scan behind the Figure 10 CFAR masking geometry (reproducibility record).

Figure 10 (``cfar_false_alarm_check.py``) shows one realization of a weak target
masked by a stronger neighbor in the same Doppler bin. The geometry used there
(weak target -15 dB at 4321.7 m, masker -10 dB nine range cells further out,
both at 101.3 m/s) was chosen with a small scan over 20 noise seeds. That scan
was originally run as a throw-away script outside the repository. This file
reproduces it exactly and saves every realization.

Scan definition (as originally run; nothing here was re-tuned):

* candidate geometries: masker offset in {5, 7, 9} range cells x masker SNR in
  {-5, -10} dB per raw sample; weak target fixed at 4321.7 m, -15 dB, phase 0.3 rad;
  masker phase 1.0 rad; both targets at 101.3 m/s;
* only these two targets are simulated. Figure 10 additionally contains a third,
  far-away target (9007.4 m, outside every CFAR window around the weak target);
* seeds 0..19, each realization drawn with ``numpy.random.default_rng(seed)``;
* processing as in Figure 10: LFM matched filter, Hann slow-time window,
  range cut at the Doppler bin nearest 101.3 m/s;
* CA-CFAR 8 training + 2 guard cells per side (N_t = 16), ``edge="valid"``;
  OS-CFAR same window, k = 12 of 16; P_FA = 1e-4;
* "detected" = any detection in the weak-target lag l0 = 288 or its two
  neighbors (l0 - 1 .. l0 + 1), the same criterion used for Figure 10.

The selected geometry is (offset 9 cells, masker -10 dB).

Usage: ``python experiments/cfar_masking_seed_scan.py``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab.cfar import ca_alpha, ca_cfar_1d, os_alpha, os_cfar_1d
from array_signal_lab.doppler import doppler_axis, doppler_process
from array_signal_lab.io import save_results
from array_signal_lab.matched_filter import matched_filter
from array_signal_lab.radar_signal import PointTarget, RadarParams, simulate_cpi
from array_signal_lab.waveforms import lfm_pulse

NAME = "cfar_masking_seed_scan"
SEEDS = list(range(20))
WEAK_RANGE_M, WEAK_SNR_DB, WEAK_PHASE = 4321.7, -15.0, 0.3
MASKER_PHASE = 1.0
VELOCITY_MPS = 101.3
OFFSETS_CELLS = (5, 7, 9)
MASKER_SNR_DB = (-5.0, -10.0)
SELECTED = (9, -10.0)
PFA = 1e-4
N_TRAIN, N_GUARD, OS_K = 8, 2, 12
WINDOW = "hann"


def run():
    t0 = time.perf_counter()
    P = RadarParams()
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    fd = doppler_axis(P.prf_hz, P.n_pulses)
    k = int(np.argmin(np.abs(fd - P.doppler_hz(VELOCITY_MPS))))
    l0 = int(round(2 * WEAK_RANGE_M / P.c * P.fs_hz))
    cells = slice(l0 - 1, l0 + 2)

    geometries, arrays = [], {"seeds": np.array(SEEDS)}
    for off in OFFSETS_CELLS:
        for m_snr in MASKER_SNR_DB:
            masker_range = WEAK_RANGE_M + off * P.range_bin
            rows = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                X = simulate_cpi(P, [PointTarget(WEAK_RANGE_M, VELOCITY_MPS, WEAK_SNR_DB, WEAK_PHASE),
                                     PointTarget(masker_range, VELOCITY_MPS, m_snr, MASKER_PHASE)], 1.0, rng)
                z = np.abs(doppler_process(matched_filter(X, h, axis=1), WINDOW)[k]) ** 2
                d_ca, t_ca = ca_cfar_1d(z, N_TRAIN, N_GUARD, PFA, "valid")
                d_os, t_os = os_cfar_1d(z, N_TRAIN, N_GUARD, OS_K, PFA)
                noise = np.median(z) / np.log(2)  # median of Exp(mean) = mean * ln 2
                rows.append({
                    "seed": seed,
                    "ca_detected": bool(d_ca[cells].any()),
                    "os_detected": bool(d_os[cells].any()),
                    "ca_threshold_at_l0": float(t_ca[l0]),
                    "os_threshold_at_l0": float(t_os[l0]),
                    "weak_cell_power_at_l0": float(z[l0]),
                    "weak_cell_power_max_l0pm1": float(z[cells].max()),
                    "noise_floor_estimate": float(noise),
                    "ca_threshold_db_re_noise": float(10 * np.log10(t_ca[l0] / noise)),
                    "os_threshold_db_re_noise": float(10 * np.log10(t_os[l0] / noise)),
                    "weak_cell_max_db_re_noise": float(10 * np.log10(z[cells].max() / noise)),
                })
            n_ca = sum(r["ca_detected"] for r in rows)
            n_os = sum(r["os_detected"] for r in rows)
            selected = (off, m_snr) == SELECTED
            geometries.append({
                "masker_offset_cells": off, "masker_range_m": masker_range, "masker_snr_db": m_snr,
                "selected_for_figure_10": selected,
                "ca_detections": n_ca, "os_detections": n_os, "n_realisations": len(rows),
                "ca_detection_fraction": n_ca / len(rows), "os_detection_fraction": n_os / len(rows),
                "per_seed": rows,
            })
            tag = f"off{off}_m{int(abs(m_snr))}"
            arrays[f"ca_detected_{tag}"] = np.array([r["ca_detected"] for r in rows])
            arrays[f"os_detected_{tag}"] = np.array([r["os_detected"] for r in rows])
            arrays[f"ca_thr_l0_{tag}"] = np.array([r["ca_threshold_at_l0"] for r in rows])
            arrays[f"os_thr_l0_{tag}"] = np.array([r["os_threshold_at_l0"] for r in rows])
            arrays[f"weak_max_{tag}"] = np.array([r["weak_cell_power_max_l0pm1"] for r in rows])

    sel = next(g for g in geometries if g["selected_for_figure_10"])
    metrics = {
        "selected_geometry": {k_: v for k_, v in sel.items() if k_ != "per_seed"},
        "selected_ca_detections": sel["ca_detections"],
        "selected_os_detections": sel["os_detections"],
        "n_realisations": len(SEEDS),
        "geometries": geometries,
        "doppler_bin_index": k,
        "weak_target_lag": l0,
        "ca_alpha": float(ca_alpha(2 * N_TRAIN, PFA)),
        "os_alpha": float(os_alpha(2 * N_TRAIN, OS_K, PFA)),
    }
    config = {
        "experiment": NAME,
        "radar": P.__dict__,
        "slow_time_window": WINDOW,
        "noise_power": 1.0,
        "weak_target": {"range_m": WEAK_RANGE_M, "velocity_mps": VELOCITY_MPS, "snr_db": WEAK_SNR_DB,
                        "phase_rad": WEAK_PHASE},
        "masker": {"velocity_mps": VELOCITY_MPS, "phase_rad": MASKER_PHASE,
                   "offset_cells_grid": OFFSETS_CELLS, "snr_db_grid": MASKER_SNR_DB,
                   "range_m": "weak range + offset * range_bin"},
        "selected_geometry": {"masker_offset_cells": SELECTED[0], "masker_snr_db": SELECTED[1]},
        "pfa": PFA,
        "ca_cfar": {"train_per_side": N_TRAIN, "guard_per_side": N_GUARD, "n_train_total": 2 * N_TRAIN,
                    "edge": "valid"},
        "os_cfar": {"train_per_side": N_TRAIN, "guard_per_side": N_GUARD, "n_train_total": 2 * N_TRAIN,
                    "k": OS_K, "edge": "valid"},
        "detection_criterion": "any detection in lags l0-1..l0+1, l0 = round(2 R_weak f_s / c)",
        "seeds": SEEDS,
        "rng": "numpy.random.default_rng(seed), one generator per realisation",
        "note": "Two targets only; Figure 10 also includes a distant third target (9007.4 m).",
    }
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    m = run()
    for g in m["geometries"]:
        print(f"offset {g['masker_offset_cells']} cells, masker {g['masker_snr_db']:+.0f} dB: "
              f"CA {g['ca_detections']}/20, OS {g['os_detections']}/20"
              + ("   <- Figure 10 geometry" if g["selected_for_figure_10"] else ""))

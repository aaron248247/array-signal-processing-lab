"""Integrated example: array reception -> beamforming -> pulse compression -> Doppler -> CA-CFAR.

Secondary to the verification studies; it reuses only verified components.

Scenario (generic): N = 16 ULA (d = lambda/2) receiving the LFM pulse train of
RadarParams. Two targets near broadside, a noise jammer at +35 deg
(JNR = 40 dB per element per sample, white over the receiver band), unit
thermal noise. Narrowband array assumption: the aperture transit time
(N-1) d sin(theta)/c is < 1 % of the range resolution 1/B, so the echo
envelope is identical across elements and only the carrier phase a(theta)
differs.

Processing: (1) conventional or MVDR weights; MVDR uses a sample covariance
from K = 20 N space-time samples drawn at random from the data cube itself
(targets occupy a small fraction of these samples); (2) beamformer output
y = w^H x for every (pulse, fast-time) sample; (3) matched filter; (4) Hann
slow-time FFT; (5) 2-D CA-CFAR, P_FA = 1e-4.

Usage: ``python experiments/integrated_array_radar.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab import ULA, conventional_weights, mvdr_weights, sample_covariance, steering_vector
from array_signal_lab.cfar import ca_cfar_2d
from array_signal_lab.covariance import complex_gaussian
from array_signal_lab.doppler import doppler_axis, doppler_process
from array_signal_lab.io import load_results, save_results
from array_signal_lab.matched_filter import matched_filter
from array_signal_lab.metrics import db, optimal_sinr, output_sinr
from array_signal_lab.radar_signal import PointTarget, RadarParams, simulate_cpi
from array_signal_lab.waveforms import lfm_pulse

NAME = "integrated_array_radar"
SEED = 20261010
P = RadarParams()
N = 16
THETA_T_DEG = [0.0, 2.0]
TARGETS = [PointTarget(4321.7, 101.3, -25.0, 0.3), PointTarget(7500.0, -60.0, -30.0, 1.7)]
JAMMER_DEG = 35.0
JNR_DB = 40.0
PFA = 1e-4
TRAIN, GUARD = (2, 8), (1, 2)
K_TRAIN = 20 * N


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    ula = ULA(N)
    a_look = steering_vector(ula, 0.0)
    # data cube (N, M, Ns)
    cube = complex_gaussian(rng, (N, P.n_pulses, P.n_fast), 1.0)
    for tg, th in zip(TARGETS, THETA_T_DEG):
        echo = simulate_cpi(P, [tg])  # noiseless, (M, Ns), per-element amplitude
        cube += steering_vector(ula, np.deg2rad(th))[:, None, None] * echo[None]
    a_j = steering_vector(ula, np.deg2rad(JAMMER_DEG))
    jam = complex_gaussian(rng, (P.n_pulses, P.n_fast), 10 ** (JNR_DB / 10))
    cube += a_j[:, None, None] * jam[None]

    X = cube.reshape(N, -1)
    idx = rng.choice(X.shape[1], K_TRAIN, replace=False)
    R_hat = sample_covariance(X[:, idx])
    R_in = np.eye(N) + 10 ** (JNR_DB / 10) * np.outer(a_j, a_j.conj())
    weights = {"conventional": conventional_weights(ula, 0.0), "mvdr": mvdr_weights(R_hat, a_look)}
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    fd = doppler_axis(P.prf_hz, P.n_pulses)
    arrays = {"range_m": P.range_axis(), "velocity_mps": fd * P.wavelength / 2}
    out = {}
    full_end = P.n_fast - h.size
    try:  # measured P_FA inflation for this processing chain (f_s = 2B, Hann), from Part E
        rec_fa, _ = load_results("cfar_false_alarm_check")
        corr_factor = next(r["ratio_measured_to_requested"] for r in rec_fa["metrics"]["pipeline"]
                           if r["case"] == "MF f_s = 2B, Hann Doppler" and r["pfa_requested"] == PFA)
    except (FileNotFoundError, StopIteration):
        corr_factor = None
    reach_r = TRAIN[1] + GUARD[1]
    for name, w in weights.items():
        y = np.tensordot(w.conj(), cube, axes=(0, 0))  # (M, Ns)
        z = np.abs(doppler_process(matched_filter(y, h, axis=1), "hann")) ** 2
        det, thr, nt = ca_cfar_2d(z, TRAIN, GUARD, PFA, wrap_axis0=True, edge="valid")
        # noise+interference floor estimated from cells away from targets
        floor = np.median(z) / np.log(2)
        found = []
        mask_targets = np.zeros_like(det)
        for tg in TARGETS:
            l0 = int(round(2 * tg.range_m / P.c * P.fs_hz))
            k0 = int(np.argmin(np.abs(fd - P.doppler_hz(tg.velocity_mps))))
            ks = [(k0 + d) % P.n_pulses for d in range(-2, 3)]
            box = np.ix_(ks, range(l0 - 3, l0 + 4))
            mask_targets[box] = True
            found.append({
                "range_m": tg.range_m, "velocity_mps": tg.velocity_mps,
                "detected": bool(det[box].any()),
                "peak_over_floor_db": float(db(z[box].max() / floor)),
            })
        valid = np.isfinite(thr)
        valid[:, full_end - reach_r:] = False  # exclude partial-overlap lags from the false-alarm count
        n_cells = int((valid & ~mask_targets).sum())
        fa_map = det & valid & ~mask_targets
        n_fa = int(fa_map.sum())
        from scipy import ndimage

        _, n_fa_clusters = ndimage.label(fa_map, structure=np.ones((3, 3)))
        s_th = [optimal_sinr(steering_vector(ula, np.deg2rad(th)), 1.0, R_in) for th in THETA_T_DEG]
        out[name] = {
            "array_output_sinr_db_per_unit_snr_target1": float(db(output_sinr(w, steering_vector(ula, 0.0), 1.0, R_in))),
            "array_output_sinr_db_per_unit_snr_target2": float(db(output_sinr(w, steering_vector(ula, np.deg2rad(THETA_T_DEG[1])), 1.0, R_in))),
            "optimal_db_per_unit_snr": [float(db(s)) for s in s_th],
            "beam_output_interference_plus_noise_power_db": float(db(np.real(np.vdot(w, R_in @ w)))),
            "measured_rd_floor_db": float(db(floor)),
            "jammer_response_db": float(20 * np.log10(np.abs(np.vdot(w, a_j)))),
            "targets": found,
            "false_alarms": n_fa, "false_alarm_clusters_8connected": int(n_fa_clusters), "cells_tested": n_cells,
            "pfa_inflation_factor_from_cfar_false_alarm_check": corr_factor,
            "expected_false_alarms_with_measured_inflation": (float(corr_factor * PFA * n_cells)
                                                              if corr_factor else None),
            "expected_false_alarms_nominal": float(PFA * n_cells),
            "n_train_cells": nt,
        }
        arrays[f"rd_{name}_db"] = (10 * np.log10(z / floor)).astype(np.float32)
        arrays[f"det_{name}"] = det
        arrays[f"floor_{name}_db"] = np.array(db(floor))
    from array_signal_lab.doppler import coherent_gain

    metrics = {
        "processing": out,
        # array gain x pulse-compression gain x Hann coherent gain, thermal noise only, grid-centered target
        "expected_post_processing_snr_db_target1_noise_only": float(
            TARGETS[0].snr_db + db(N) + db(P.n_pulse_samples) + db(coherent_gain("hann", P.n_pulses))),
    }
    config = {"N": N, "targets": [t.__dict__ for t in TARGETS], "target_angles_deg": THETA_T_DEG,
              "jammer_deg": JAMMER_DEG, "jnr_db": JNR_DB, "pfa": PFA, "train": TRAIN, "guard": GUARD,
              "K_train": K_TRAIN, "radar": P.__dict__, "seed": SEED}
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import save, setup

    setup()
    rec, z = load_results(NAME)
    m = rec["metrics"]["processing"]
    r, v = z["range_m"] / 1e3, z["velocity_mps"]
    dv = v[1] - v[0]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for ax, name, lab in ((axes[0], "conventional", "(a) Conventional beam"), (axes[1], "mvdr", "(b) MVDR beam (K = 20N)")):
        rd = z[f"rd_{name}_db"].astype(float)
        pool = 5
        nr = (rd.shape[1] // pool) * pool
        pooled = rd[:, :nr].reshape(rd.shape[0], -1, pool).max(axis=2)
        im = ax.imshow(pooled, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=25,
                       interpolation="nearest", extent=[r[0], r[nr - 1], v[0] - dv / 2, v[-1] + dv / 2])
        kk, ll = np.where(z[f"det_{name}"])
        ax.plot(r[ll], v[kk], "s", ms=6, mfc="none", mec="#e34948", mew=1.2, label="CA-CFAR detections")
        for t in rec["config"]["targets"]:
            ax.plot(t["range_m"] / 1e3, t["velocity_mps"], "o", ms=13, mfc="none", mec="white", mew=1.0)
        floor = float(z[f"floor_{name}_db"])
        ax.set_title(f"{lab}: floor {floor:.1f} dB re: element noise, {m[name]['false_alarms']} false-alarm cells", loc="left")
        ax.set_xlabel("range (km)")
        ax.set_ylabel("radial velocity (m/s)")
        ax.grid(False)
        ax.legend(loc="lower right", fontsize=7.5, facecolor="white", frameon=True, framealpha=0.85)
    cb = fig.colorbar(im, ax=axes, pad=0.01)
    cb.set_label("power re: each map's own noise floor (dB)")
    return save(fig, "fig13_integrated_array_radar.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        print(json.dumps(run(), indent=1))
    print(plot())

"""Part C - steering-vector mismatch and target self-nulling.

The beamformer is steered to theta_0 = 0 deg while the target actually arrives
from theta_0 + delta. Two regimes are distinguished:

* MVDR built from the interference-plus-noise covariance R_in (target absent
  from the training data): loss = cos^2(a0, a_s; R_in^{-1}).
* MPDR, i.e. the same formula applied to a covariance that *contains* the
  target: the target outside the exact look direction is treated as
  interference and partially nulled. Closed form [Cox 1973]
  SINR = S c2 / (1 + (2S + S^2)(1 - c2)).

Finite-sample SMI with the target in the training data, with and without
diagonal loading, is evaluated by Monte Carlo.

Usage: ``python experiments/steering_mismatch.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab import (
    conventional_weights,
    mvdr_weights,
    response,
    steering_vector,
    theoretical_covariance,
)
from array_signal_lab.io import load_results, save_results
from array_signal_lab.metrics import (
    db,
    mismatched_mpdr_sinr,
    mismatched_mvdr_sinr,
    optimal_sinr,
    output_sinr,
)
from common import JAMMERS_DEG, JNR_DB, N, baseline, mean_ci_db, smi_trials

NAME = "steering_mismatch"
SEED = 20261005
N_TRIALS = 300
DELTA_DEG = np.linspace(-4.0, 4.0, 33)
SNR_SWEEP_DB = np.arange(-10.0, 20.1, 2.5)
DELTA_FIXED_DEG = 1.0
K_SMI = 10 * N
LOADINGS_DB = (10.0, 20.0)


def _set_snr(sc, snr_db):
    sc = dict(sc)
    sc["snr"] = 10 ** (snr_db / 10)
    return sc


def _deterministic(sc, a0, th_true):
    ula = sc["ula"]
    a_s = steering_vector(ula, th_true)
    s_opt = optimal_sinr(a_s, sc["snr"], sc["R_in"])
    wc = conventional_weights(ula, 0.0)
    return {
        "conventional": db(output_sinr(wc, a_s, sc["snr"], sc["R_in"]) / s_opt),
        "mvdr_Rin": db(mismatched_mvdr_sinr(a0, a_s, sc["snr"], sc["R_in"]) / s_opt),
        "mpdr_R": db(mismatched_mpdr_sinr(a0, a_s, sc["snr"], sc["R_in"]) / s_opt),
    }


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    base = baseline()
    ula = base["ula"]
    a0 = steering_vector(ula, 0.0)
    arrays = {"delta_deg": DELTA_DEG, "snr_sweep_db": SNR_SWEEP_DB}

    # (a) sweep delta at SNR = 0 dB per element
    sc = _set_snr(base, 0.0)
    det = {k: [] for k in ("conventional", "mvdr_Rin", "mpdr_R")}
    mc = {"smi": [], **{f"smi_load{int(l)}": [] for l in LOADINGS_DB}}
    for d in DELTA_DEG:
        th = np.deg2rad(d)
        for k, v in _deterministic(sc, a0, th).items():
            det[k].append(v)
        rho, _, _ = smi_trials(sc, K_SMI, N_TRIALS, rng, signal_in_training=True, a_look=a0, th_s_true=th)
        mc["smi"].append(mean_ci_db(rho)[0])
        for l in LOADINGS_DB:
            rho, _, _ = smi_trials(sc, K_SMI, N_TRIALS, rng, signal_in_training=True, a_look=a0,
                                   th_s_true=th, loading=10 ** (l / 10))
            mc[f"smi_load{int(l)}"].append(mean_ci_db(rho)[0])
    for k, v in {**det, **mc}.items():
        arrays[f"delta_{k}_db"] = np.array(v)

    # (b) sweep SNR at fixed delta
    th = np.deg2rad(DELTA_FIXED_DEG)
    det = {k: [] for k in ("conventional", "mvdr_Rin", "mpdr_R")}
    mc = {"smi": [], **{f"smi_load{int(l)}": [] for l in LOADINGS_DB}}
    for snr_db in SNR_SWEEP_DB:
        sc = _set_snr(base, snr_db)
        for k, v in _deterministic(sc, a0, th).items():
            det[k].append(v)
        rho, _, _ = smi_trials(sc, K_SMI, N_TRIALS, rng, signal_in_training=True, a_look=a0, th_s_true=th)
        mc["smi"].append(mean_ci_db(rho)[0])
        for l in LOADINGS_DB:
            rho, _, _ = smi_trials(sc, K_SMI, N_TRIALS, rng, signal_in_training=True, a_look=a0,
                                   th_s_true=th, loading=10 ** (l / 10))
            mc[f"smi_load{int(l)}"].append(mean_ci_db(rho)[0])
    for k, v in {**det, **mc}.items():
        arrays[f"snr_{k}_db"] = np.array(v)

    # (c) beampatterns, delta = 1 deg, SNR = 10 dB, clairvoyant
    sc = _set_snr(base, 10.0)
    a_s = steering_vector(ula, th)
    R = sc["R_in"] + sc["snr"] * np.outer(a_s, a_s.conj())
    thetas = np.deg2rad(np.linspace(-15, 15, 1201))
    arrays["bp_theta_deg"] = np.rad2deg(thetas)
    arrays["bp_conventional"] = np.abs(response(conventional_weights(ula, 0.0), ula, thetas))
    arrays["bp_mvdr_Rin"] = np.abs(response(mvdr_weights(sc["R_in"], a0), ula, thetas))
    arrays["bp_mpdr_R"] = np.abs(response(mvdr_weights(R, a0), ula, thetas))
    arrays["bp_mpdr_R_load20"] = np.abs(response(mvdr_weights(R, a0, loading=100.0), ula, thetas))
    w_mpdr = mvdr_weights(R, a0)
    gain_true_dir_db = float(20 * np.log10(np.abs(response(w_mpdr, ula, th))))
    # beam-pointing: location of the pattern maximum within the main lobe
    i_pk = int(np.argmax(arrays["bp_mpdr_R"][np.abs(arrays["bp_theta_deg"]) < 5]))
    pk_deg = float(arrays["bp_theta_deg"][np.abs(arrays["bp_theta_deg"]) < 5][i_pk])

    def at(arr_key, grid, value):
        return float(arrays[arr_key][int(np.argmin(np.abs(grid - value)))])

    metrics = {
        "delta_sweep_snr0dB_at_delta_1deg": {
            k: at(f"delta_{k}_db", DELTA_DEG, 1.0)
            for k in ("conventional", "mvdr_Rin", "mpdr_R", "smi", "smi_load10", "smi_load20")
        },
        "delta_sweep_snr0dB_at_delta_2p5deg": {
            k: at(f"delta_{k}_db", DELTA_DEG, 2.5)
            for k in ("conventional", "mvdr_Rin", "mpdr_R", "smi", "smi_load10", "smi_load20")
        },
        "snr_sweep_delta1deg": {
            f"snr{s:+.0f}dB": {k: float(arrays[f"snr_{k}_db"][i])
                               for k in ("conventional", "mvdr_Rin", "mpdr_R", "smi", "smi_load10", "smi_load20")}
            for i, s in enumerate(SNR_SWEEP_DB) if s % 10 == 0
        },
        "mpdr_snr10_delta1_response_toward_true_target_db": gain_true_dir_db,
        "mpdr_snr10_delta1_mainlobe_peak_deg": pk_deg,
    }
    config = {
        "N": N, "jammers_deg": JAMMERS_DEG, "jnr_db": JNR_DB, "look_deg": 0.0,
        "delta_deg_grid": DELTA_DEG, "snr_sweep_db": SNR_SWEEP_DB, "delta_fixed_deg": DELTA_FIXED_DEG,
        "K_smi": K_SMI, "loadings_db": LOADINGS_DB, "n_trials_per_point": N_TRIALS, "seed": SEED,
    }
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, INK, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    series = [
        ("conventional", "conventional", C[0], "-", None),
        ("mvdr_Rin", r"MVDR, true $R_{in}$ (closed form)", C[2], "-", None),
        ("mpdr_R", r"MPDR, true $R$ incl. target (Cox)", C[1], "-", None),
        ("smi", "SMI, K = 10N, target in training", C[1], "none", "o"),
        ("smi_load10", "  + 10 dB loading", C[6], "none", "^"),
        ("smi_load20", "  + 20 dB loading", C[3], "none", "s"),
    ]
    for ax, prefix, x, xl, title in [
        (axes[0], "delta", z["delta_deg"], r"pointing error $\delta = \theta_s - \theta_0$ (deg)",
         "(a) SINR loss vs pointing error, SNR = 0 dB/element"),
        (axes[1], "snr", z["snr_sweep_db"], "per-element SNR (dB)",
         r"(b) SINR loss vs SNR, $\delta$ = 1°"),
    ]:
        for key, lab, col, ls, mk in series:
            ax.plot(x, z[f"{prefix}_{key}_db"], color=col, ls=ls, marker=mk, ms=4, mfc="none",
                    lw=1.5 if ls != "none" else 0, label=lab)
        ax.set_xlabel(xl)
        ax.set_ylabel(r"SINR / SINR$_{\rm opt}(\theta_s)$ (dB)")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(-45, 2)
    axes[1].set_ylim(-45, 2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.36, -0.02))

    ax = axes[2]
    th = z["bp_theta_deg"]
    for key, lab, col, ls in [("bp_conventional", "conventional", C[0], "-"),
                              ("bp_mvdr_Rin", r"MVDR, $R_{in}$", C[2], "--"),
                              ("bp_mpdr_R", r"MPDR, $R$", C[1], "-"),
                              ("bp_mpdr_R_load20", r"MPDR, $R$ + 20 dB loading", C[3], ":")]:
        ax.plot(th, 20 * np.log10(np.maximum(z[key], 1e-6)), color=col, ls=ls, label=lab)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(rec["config"]["delta_fixed_deg"], color=INK, lw=0.8, ls="-.")
    ax.text(1.3, -28, "true target", fontsize=7.5, color=INK)
    ax.text(-0.3, -28, "look", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-50, 5)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$20\log_{10}|w^H a(\theta)|$ (dB)")
    ax.set_title(r"(c) Patterns, $\delta$ = 1°, SNR = 10 dB, true covariances", loc="left")
    ax.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor="white", edgecolor="white",
              framealpha=0.95)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, "fig05_steering_mismatch.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        print(json.dumps(run(), indent=1))
    print(plot())

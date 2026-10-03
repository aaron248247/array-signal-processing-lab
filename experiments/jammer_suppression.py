"""Part B - conventional vs MVDR beamforming with two sidelobe jammers.

Scenario (generic): N = 16, d = lambda/2, target at 0 deg with per-element
SNR 0 dB, jammers at +35 deg (JNR 40 dB) and -42 deg (JNR 30 dB), unit noise.

Beamformers compared
--------------------
* conventional ``a_s / N``
* MVDR with the true interference-plus-noise covariance (clairvoyant)
* MVDR from the sample covariance (SMI) of K signal-free snapshots, K = 2N, 10N
* SMI with the target present in the training data, K = 10N
* SMI, K = 2N, diagonal loading 10 dB above the noise level

Metrics: target response, jammer response relative to target, output SINR
(evaluated with the true R_in), SINR loss relative to the optimum, white-noise
gain. SMI metrics are Monte Carlo means over independent training sets and
are compared with the Reed-Mallett-Brennan (RMB) prediction where it applies.

Usage: ``python experiments/jammer_suppression.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab import (
    ULA,
    conventional_weights,
    mvdr_weights,
    response,
    sample_covariance,
    simulate_snapshots,
    steering_vector,
    theoretical_covariance,
)
from array_signal_lab.io import load_results, save_results
from array_signal_lab.metrics import (
    db,
    optimal_sinr,
    output_sinr,
    relative_response_db,
    rmb_mean_normalized_sinr,
    smi_signal_present_mean_normalized_sinr,
    white_noise_gain,
)

NAME = "jammer_suppression"
SEED = 20261002
N = 16
THETA_S = 0.0
JAMMERS_DEG = [35.0, -42.0]
JNR_DB = [40.0, 30.0]
SNR_DB = 0.0
N_TRIALS = 2000


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    ula = ULA(N)
    th_s = np.deg2rad(THETA_S)
    th_j = np.deg2rad(JAMMERS_DEG)
    jnr = 10 ** (np.array(JNR_DB) / 10)
    snr = 10 ** (SNR_DB / 10)
    a_s = steering_vector(ula, th_s)
    A_j = steering_vector(ula, th_j)
    R_in = theoretical_covariance(ula, th_j, jnr, 1.0)
    R_full = theoretical_covariance(ula, np.r_[th_s, th_j], np.r_[snr, jnr], 1.0)
    sinr_opt = optimal_sinr(a_s, snr, R_in)

    def summarize(w):
        return {
            "target_gain_db": float(20 * np.log10(np.abs(np.vdot(w, a_s)))),
            "jammer_rel_db": [relative_response_db(w, a_s, A_j[:, i]) for i in range(len(th_j))],
            "sinr_db": float(db(output_sinr(w, a_s, snr, R_in))),
            "sinr_loss_db": float(db(output_sinr(w, a_s, snr, R_in) / sinr_opt)),
            "wng_db": float(db(white_noise_gain(w, a_s))),
        }

    w_conv = conventional_weights(ula, th_s)
    w_clair = mvdr_weights(R_in, a_s)
    w_clair_full = mvdr_weights(R_full, a_s)  # signal-included true covariance
    deterministic = {
        "conventional": summarize(w_conv),
        "mvdr_clairvoyant_Rin": summarize(w_clair),
        "mvdr_clairvoyant_Rfull": summarize(w_clair_full),
        "max_abs_diff_w_Rin_vs_Rfull": float(np.max(np.abs(w_clair - w_clair_full))),
    }

    smi_cases = [
        ("smi_K32_signal_free", 2 * N, False, 0.0),
        ("smi_K160_signal_free", 10 * N, False, 0.0),
        ("smi_K160_signal_present", 10 * N, True, 0.0),
        ("smi_K32_loaded_10dB", 2 * N, False, 10.0),
    ]
    mc = {}
    example_w = {}
    for name, k, with_sig, lnr in smi_cases:
        rho, jr0, jr1, tg = [], [], [], []
        for trial in range(N_TRIALS):
            if with_sig:
                X = simulate_snapshots(ula, np.r_[th_s, th_j], np.r_[snr, jnr], 1.0, k, rng)
            else:
                X = simulate_snapshots(ula, th_j, jnr, 1.0, k, rng)
            w = mvdr_weights(sample_covariance(X), a_s, loading=lnr and 10 ** (lnr / 10))
            if trial == 0:
                example_w[name] = w
            rho.append(output_sinr(w, a_s, snr, R_in) / sinr_opt)
            jr0.append(relative_response_db(w, a_s, A_j[:, 0]))
            jr1.append(relative_response_db(w, a_s, A_j[:, 1]))
            tg.append(20 * np.log10(np.abs(np.vdot(w, a_s))))
        rho = np.array(rho)
        mc[name] = {
            "K": k,
            "signal_in_training": with_sig,
            "loading_db_rel_noise": lnr if lnr else None,
            "rho_mean": float(rho.mean()),
            "rho_mean_stderr": float(rho.std(ddof=1) / np.sqrt(N_TRIALS)),
            "rho_mean_db": float(db(rho.mean())),
            "rho_p05_db": float(db(np.percentile(rho, 5))),
            "rho_p95_db": float(db(np.percentile(rho, 95))),
            "rmb_mean_prediction": rmb_mean_normalized_sinr(k, N) if (not with_sig and not lnr) else None,
            "signal_present_mean_prediction": (
                smi_signal_present_mean_normalized_sinr(k, N, sinr_opt) if (with_sig and not lnr) else None
            ),
            "jammer1_rel_db_median": float(np.median(jr0)),
            "jammer2_rel_db_median": float(np.median(jr1)),
            "target_gain_db_max_abs": float(np.max(np.abs(tg))),
        }

    thetas = np.deg2rad(np.linspace(-90, 90, 3601))
    arrays = {
        "theta_deg": np.rad2deg(thetas),
        "bp_conventional": np.abs(response(w_conv, ula, thetas)),
        "bp_mvdr_clairvoyant": np.abs(response(w_clair, ula, thetas)),
    }
    for name, w in example_w.items():
        arrays[f"bp_{name}"] = np.abs(response(w, ula, thetas))

    metrics = {
        "sinr_opt_db": float(db(sinr_opt)),
        "snr_array_db_no_jammer": float(db(snr * N)),
        "deterministic": deterministic,
        "monte_carlo": mc,
    }
    config = {
        "N": N, "d_over_lambda": 0.5, "theta_s_deg": THETA_S, "snr_db_per_element": SNR_DB,
        "jammers_deg": JAMMERS_DEG, "jnr_db_per_element": JNR_DB, "noise_power": 1.0,
        "n_trials": N_TRIALS, "seed": SEED,
        "smi_cases": [{"name": c[0], "K": c[1], "signal_in_training": c[2], "loading_db": c[3]} for c in smi_cases],
    }
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    m = rec["metrics"]
    th = z["theta_deg"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    curves = [
        ("bp_conventional", "conventional", C[0], "-"),
        ("bp_mvdr_clairvoyant", "MVDR, true $R_{in}$", C[1], "-"),
        ("bp_smi_K32_signal_free", "SMI, K = 2N (one draw)", C[2], "--"),
        ("bp_smi_K32_loaded_10dB", "SMI, K = 2N, 10 dB loading (one draw)", C[6], ":"),
    ]
    for key, lab, col, ls in curves:
        ax.plot(th, 20 * np.log10(np.maximum(z[key], 1e-6)), color=col, ls=ls, label=lab, lw=1.4)
    for jd, jn in zip(rec["config"]["jammers_deg"], rec["config"]["jnr_db_per_element"]):
        ax.axvline(jd, color=MUTED, lw=0.8, ls="-.")
        ax.text(jd + 1, 3, f"jammer\n{jn:.0f} dB", fontsize=7.5, color=MUTED, va="top")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.text(1, 3, "target", fontsize=7.5, color=MUTED, va="top")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-100, 5)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$20\log_{10}|w^H a(\theta)|$ (dB, re: target response)")
    ax.set_title("(a) Beampatterns, N = 16", loc="left")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    labels, vals, lo, hi, pred = [], [], [], [], []
    d = m["deterministic"]
    labels += ["conventional", "MVDR, true $R_{in}$"]
    vals += [d["conventional"]["sinr_loss_db"], d["mvdr_clairvoyant_Rin"]["sinr_loss_db"]]
    lo += [0, 0]
    hi += [0, 0]
    pred += [None, 0.0]
    names = {
        "smi_K32_signal_free": "SMI K=2N",
        "smi_K160_signal_free": "SMI K=10N",
        "smi_K160_signal_present": "SMI K=10N,\ntarget in training",
        "smi_K32_loaded_10dB": "SMI K=2N,\n10 dB loading",
    }
    for key, lab in names.items():
        r = m["monte_carlo"][key]
        labels.append(lab)
        vals.append(r["rho_mean_db"])
        lo.append(r["rho_mean_db"] - r["rho_p05_db"])
        hi.append(r["rho_p95_db"] - r["rho_mean_db"])
        p = r["rmb_mean_prediction"] or r["signal_present_mean_prediction"]
        pred.append(db(p) if p else None)
    y = np.arange(len(labels))[::-1]
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="o", color=C[0], ms=6, capsize=3,
                label="value (SMI: MC mean, 5–95 % range)")
    py = [yy for yy, p in zip(y, pred) if p is not None]
    pv = [p for p in pred if p is not None]
    ax.plot(pv, py, ls="none", marker="D", mfc="none", mec=C[1], ms=9, mew=1.5,
            label="theory (optimum; RMB mean;\nsignal-present mean)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"SINR loss re: optimum, $10\log_{10}(\mathrm{SINR}/\mathrm{SINR}_{\rm opt})$ (dB)")
    ax.set_title(f"(b) Output SINR loss (SINR$_{{opt}}$ = {m['sinr_opt_db']:.2f} dB)", loc="left")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig02_conventional_vs_mvdr.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        print(json.dumps(run(), indent=1))
    print(plot())

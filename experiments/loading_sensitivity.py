"""Part C - diagonal-loading sensitivity of the SMI/MVDR beamformer.

Loading ``R_hat + delta I`` is swept over ``delta / sigma^2`` = -30 ... +60 dB
for the baseline scenario. Expected limiting behaviour:

* ``delta -> 0``: unloaded SMI (RMB loss for signal-free training).
* ``delta >> lambda_max(R)``: weights -> conventional ``a_s / N``; the jammers
  are no longer suppressed and the SINR tends to the conventional value.
* in between, loading suppresses the noise-eigenvalue fluctuations that cause
  the small-K loss [Carlson 1988]; it also limits target self-nulling when
  the target is present in the training data.

A clairvoyant curve (true ``R_in + delta I``) isolates the bias introduced by
loading from the estimation error.

Usage: ``python experiments/loading_sensitivity.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab import conventional_weights, mvdr_weights, steering_vector
from array_signal_lab.io import load_results, save_results
from array_signal_lab.metrics import db, output_sinr, relative_response_db
from common import JAMMERS_DEG, JNR_DB, N, SNR_DB, baseline, mean_ci_db, smi_trials

NAME = "loading_sensitivity"
SEED = 20261004
N_TRIALS = 400
LNR_DB = np.arange(-30.0, 60.1, 2.5)


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    sc = baseline()
    ula, a_s = sc["ula"], sc["a_s"]
    A_j = steering_vector(ula, sc["th_j"])
    cases = {
        "K8_signal_free": dict(k=8, signal_in_training=False),
        "K32_signal_free": dict(k=32, signal_in_training=False),
        "K32_signal_present": dict(k=32, signal_in_training=True),
        "K160_signal_present": dict(k=160, signal_in_training=True),
    }
    arrays = {"lnr_db": LNR_DB}
    for name, kw in cases.items():
        m_, lo_, hi_, j1, j2 = [], [], [], [], []
        for lnr in LNR_DB:
            rho, _, jr = smi_trials(
                sc, kw["k"], N_TRIALS, rng, signal_in_training=kw["signal_in_training"],
                loading=10 ** (lnr / 10), extra=lambda w: relative_response_db(w, a_s, A_j[:, 0]))
            m, lo, hi = mean_ci_db(rho)
            m_.append(m); lo_.append(lo); hi_.append(hi)
            j1.append(np.median(jr))
        arrays[f"{name}_mean_db"] = np.array(m_)
        arrays[f"{name}_lo_db"] = np.array(lo_)
        arrays[f"{name}_hi_db"] = np.array(hi_)
        arrays[f"{name}_jammer1_rel_db_median"] = np.array(j1)
    # clairvoyant loaded MVDR (deterministic)
    cl, cl_j1, cl_j2 = [], [], []
    for lnr in LNR_DB:
        w = mvdr_weights(sc["R_in"], a_s, loading=10 ** (lnr / 10))
        cl.append(db(output_sinr(w, a_s, sc["snr"], sc["R_in"]) / sc["sinr_opt"]))
        cl_j1.append(relative_response_db(w, a_s, A_j[:, 0]))
        cl_j2.append(relative_response_db(w, a_s, A_j[:, 1]))
    arrays["clairvoyant_db"] = np.array(cl)
    arrays["clairvoyant_jammer1_rel_db"] = np.array(cl_j1)
    arrays["clairvoyant_jammer2_rel_db"] = np.array(cl_j2)
    wc = conventional_weights(ula, sc["th_s"])
    conv_db = float(db(output_sinr(wc, a_s, sc["snr"], sc["R_in"]) / sc["sinr_opt"]))
    ev = np.linalg.eigvalsh(sc["R_in"])

    best = {}
    for name in cases:
        m = arrays[f"{name}_mean_db"]
        i = int(np.nanargmax(m))
        # range of loading within 1 dB of the best mean
        good = LNR_DB[m >= m[i] - 1.0]
        i_lo = int(np.argmin(np.abs(LNR_DB - (-30.0))))
        best[name] = {
            "best_lnr_db": float(LNR_DB[i]),
            "best_mean_rho_db": float(m[i]),
            "lnr_range_within_1dB_of_best": [float(good.min()), float(good.max())],
            "mean_rho_db_at_lnr_-30": float(m[i_lo]) if np.isfinite(m[i_lo]) else None,
            "mean_rho_db_at_lnr_+60": float(m[-1]),
        }
    metrics = {
        "sinr_opt_db": float(db(sc["sinr_opt"])),
        "conventional_rho_db": conv_db,
        "eigenvalues_Rin_top2_db": [float(db(ev[-1])), float(db(ev[-2]))],
        "per_case": best,
        "clairvoyant_rho_db_at_lnr": {f"{l:+.1f}": float(v) for l, v in zip(LNR_DB, cl) if l % 10 == 0},
    }
    config = {
        "N": N, "jammers_deg": JAMMERS_DEG, "jnr_db": JNR_DB, "snr_db_per_element": SNR_DB,
        "lnr_db_grid": LNR_DB, "n_trials_per_point": N_TRIALS, "cases": cases, "seed": SEED,
    }
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    m = rec["metrics"]
    L = z["lnr_db"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]
    styles = [
        ("K8_signal_free", "K = N/2, signal-free", C[3], "D"),
        ("K32_signal_free", "K = 2N, signal-free", C[0], "o"),
        ("K32_signal_present", "K = 2N, target in training", C[1], "s"),
        ("K160_signal_present", "K = 10N, target in training", C[6], "v"),
    ]
    for key, lab, col, mk in styles:
        y = z[f"{key}_mean_db"]
        ax.plot(L, y, marker=mk, ms=4, mfc="none", color=col, lw=1.0, label=lab)
    ax.plot(L, z["clairvoyant_db"], color=INK_COLOR, lw=1.6, ls="--", label=r"true $R_{in}$ + loading")
    ax.axhline(m["conventional_rho_db"], color=MUTED, lw=0.9, ls=":")
    ax.text(-29, m["conventional_rho_db"] + 0.5, "conventional beamformer", fontsize=7.5, color=MUTED)
    for evdb in m["eigenvalues_Rin_top2_db"]:
        ax.axvline(evdb, color=MUTED, lw=0.7, ls="-.")
    ax.text(m["eigenvalues_Rin_top2_db"][0] + 0.5, -14, r"$\lambda_1,\lambda_2$ of $R_{in}$", fontsize=7.5,
            color=MUTED, rotation=90)
    ax.set_ylim(-32, 1)
    ax.set_xlabel(r"loading level $\delta/\sigma^2$ (dB)")
    ax.set_ylabel(r"mean $\rho$ = SINR / SINR$_{\rm opt}$ (dB)")
    ax.set_title("(a) Output SINR vs diagonal loading", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), fontsize=7.5)

    ax = axes[1]
    ax.plot(L, z["K32_signal_free_jammer1_rel_db_median"], color=C[0], marker="o", ms=4, mfc="none", lw=1.0,
            label="K = 2N, signal-free (median over trials)")
    ax.plot(L, z["K32_signal_present_jammer1_rel_db_median"], color=C[1], marker="s", ms=4, mfc="none", lw=1.0,
            label="K = 2N, target in training (median)")
    ax.plot(L, z["clairvoyant_jammer1_rel_db"], color=INK_COLOR, ls="--", lw=1.6, label=r"true $R_{in}$ + loading")
    ax.set_xlabel(r"loading level $\delta/\sigma^2$ (dB)")
    ax.set_ylabel("jammer 1 (35°, 40 dB) response re: target (dB)")
    ax.set_title("(b) Jammer-1 suppression vs loading", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig04_loading_sensitivity.png")


INK_COLOR = "#0b0b0b"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        print(json.dumps(run(), indent=1))
    print(plot())

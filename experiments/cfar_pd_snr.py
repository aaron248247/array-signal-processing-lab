"""Part E - probability of detection versus SNR for CA-CFAR at fixed P_FA.

Single-cell detection after square-law detection, unit noise power, N_t = 16
training cells (8 per side, 2 guard cells per side), P_FA = 1e-4.

Theory curves
-------------
* ideal known-noise detector, non-fluctuating target (Swerling 0):
  Marcum Q, ``Q1(sqrt(2 SNR), sqrt(-2 ln P_FA))`` [Richards 2014, ch. 6].
* ideal detector, Swerling 1: ``P_FA^(1/(1+SNR))``.
* CA-CFAR, Swerling 1: ``(1 + alpha / (N_t (1 + SNR)))^(-N_t)`` [Finn & Johnson 1968].
* CA-CFAR, Swerling 0: quadrature over the Gamma-distributed noise estimate.

Monte Carlo: 20 000 independent windows per SNR point (independent windows,
so the detection count is exactly binomial; Clopper-Pearson 95 % intervals).
An independent cross-check of the ideal Swerling-0 curve uses Albersheim's
approximation [Albersheim 1981].

Usage: ``python experiments/cfar_pd_snr.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab.cfar import (
    binomial_ci,
    ca_cfar_1d,
    pd_ca_swerling0,
    pd_ca_swerling1,
    pd_fixed_threshold_swerling0,
    pd_fixed_threshold_swerling1,
    required_snr_db,
)
from array_signal_lab.covariance import complex_gaussian
from array_signal_lab.io import load_results, save_results

NAME = "cfar_pd_snr"
SEED = 20261009
PFA = 1e-4
N_TRAIN_SIDE, N_GUARD_SIDE = 8, 2
NT = 2 * N_TRAIN_SIDE
SNR_THEORY_DB = np.linspace(0.0, 30.0, 151)
SNR_MC_DB = np.r_[np.arange(4.0, 16.1, 1.0), np.arange(18.0, 28.1, 2.0)]
N_TRIALS = 20_000
NT_SWEEP = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])


def albersheim_snr_db(pd, pfa, n=1):
    """Albersheim (1981) empirical SNR (dB) for a non-fluctuating target and n noncoherently
    integrated pulses (linear detector); quoted accuracy ~0.2 dB for 1e-7 < pfa < 1e-3, 0.1 < pd < 0.9.

        SNR_dB = -5 log10 n + (6.2 + 4.54 / sqrt(n + 0.44)) log10(A + 0.12 A B + 1.7 B),
        A = ln(0.62 / pfa),  B = ln(pd / (1 - pd)).
    """
    a = np.log(0.62 / pfa)
    b = np.log(pd / (1 - pd))
    return -5 * np.log10(n) + (6.2 + 4.54 / np.sqrt(n + 0.44)) * np.log10(a + 0.12 * a * b + 1.7 * b)


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    snr = 10 ** (SNR_THEORY_DB / 10)
    arrays = {"snr_theory_db": SNR_THEORY_DB, "snr_mc_db": SNR_MC_DB, "nt_sweep": NT_SWEEP}
    arrays["th_ideal_sw0"] = np.array([pd_fixed_threshold_swerling0(s, PFA) for s in snr])
    arrays["th_ideal_sw1"] = np.array([pd_fixed_threshold_swerling1(s, PFA) for s in snr])
    arrays["th_ca_sw0"] = np.array([pd_ca_swerling0(s, PFA, NT) for s in snr])
    arrays["th_ca_sw1"] = np.array([pd_ca_swerling1(s, PFA, NT) for s in snr])

    width = 2 * (N_TRAIN_SIDE + N_GUARD_SIDE) + 1
    cut = N_TRAIN_SIDE + N_GUARD_SIDE
    mc = {"sw0": [], "sw1": []}
    rows = []
    for s_db in SNR_MC_DB:
        s = 10 ** (s_db / 10)
        row = {"snr_db": float(s_db)}
        for model in ("sw0", "sw1"):
            noise = complex_gaussian(rng, (N_TRIALS, width), 1.0)
            if model == "sw0":
                amp = np.sqrt(s) * np.exp(1j * rng.uniform(0, 2 * np.pi, N_TRIALS))
            else:
                amp = complex_gaussian(rng, N_TRIALS, s)
            x = noise
            x[:, cut] += amp
            det, _ = ca_cfar_1d(np.abs(x) ** 2, N_TRAIN_SIDE, N_GUARD_SIDE, PFA, "valid")
            k = int(det[:, cut].sum())
            lo, hi = binomial_ci(k, N_TRIALS, 0.95)
            th = pd_ca_swerling0(s, PFA, NT) if model == "sw0" else pd_ca_swerling1(s, PFA, NT)
            mc[model].append((k / N_TRIALS, lo, hi))
            row[model] = {"pd_mc": k / N_TRIALS, "ci95": [lo, hi], "pd_theory": th, "theory_in_ci": bool(lo <= th <= hi)}
        rows.append(row)
    for model in ("sw0", "sw1"):
        a = np.array(mc[model])
        arrays[f"mc_{model}"] = a[:, 0]
        arrays[f"mc_{model}_lo"] = a[:, 1]
        arrays[f"mc_{model}_hi"] = a[:, 2]

    # required SNR and CFAR loss
    req = {}
    for pd in (0.5, 0.9):
        r = {
            "ideal_sw0_db": required_snr_db(pd, pd_fixed_threshold_swerling0, PFA),
            "ca16_sw0_db": required_snr_db(pd, pd_ca_swerling0, PFA, NT),
            "ideal_sw1_db": required_snr_db(pd, pd_fixed_threshold_swerling1, PFA),
            "ca16_sw1_db": required_snr_db(pd, pd_ca_swerling1, PFA, NT),
            "albersheim_sw0_db": float(albersheim_snr_db(pd, PFA)),
        }
        r["cfar_loss_sw0_db"] = r["ca16_sw0_db"] - r["ideal_sw0_db"]
        r["cfar_loss_sw1_db"] = r["ca16_sw1_db"] - r["ideal_sw1_db"]
        r["fluctuation_loss_ideal_db"] = r["ideal_sw1_db"] - r["ideal_sw0_db"]
        req[f"pd{pd}"] = r
    loss05, loss09 = [], []
    for nt in NT_SWEEP:
        loss05.append(required_snr_db(0.5, pd_ca_swerling0, PFA, int(nt)) - req["pd0.5"]["ideal_sw0_db"])
        loss09.append(required_snr_db(0.9, pd_ca_swerling0, PFA, int(nt)) - req["pd0.9"]["ideal_sw0_db"])
    arrays["cfar_loss_pd05_db"] = np.array(loss05)
    arrays["cfar_loss_pd09_db"] = np.array(loss09)

    n_in = sum(r[m]["theory_in_ci"] for r in rows for m in ("sw0", "sw1"))
    metrics = {
        "pfa": PFA, "n_train": NT,
        "mc_table": rows,
        "theory_inside_95ci_count": int(n_in), "n_mc_points": 2 * len(rows),
        "required_snr": req,
        "cfar_loss_vs_nt_pd05_db": dict(zip([int(n) for n in NT_SWEEP], [float(v) for v in loss05])),
        "cfar_loss_vs_nt_pd09_db": dict(zip([int(n) for n in NT_SWEEP], [float(v) for v in loss09])),
    }
    config = {"pfa": PFA, "n_train_per_side": N_TRAIN_SIDE, "n_guard_per_side": N_GUARD_SIDE,
              "snr_mc_db": SNR_MC_DB, "n_trials": N_TRIALS, "seed": SEED}
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    m = rec["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = z["snr_theory_db"]
    ax.plot(x, z["th_ideal_sw0"], color=C[0], ls=":", lw=1.3, label="ideal (known noise), Swerling 0")
    ax.plot(x, z["th_ca_sw0"], color=C[0], lw=1.6, label="CA-CFAR N$_t$=16, Swerling 0 (quadrature)")
    ax.plot(x, z["th_ideal_sw1"], color=C[1], ls=":", lw=1.3, label="ideal (known noise), Swerling 1")
    ax.plot(x, z["th_ca_sw1"], color=C[1], lw=1.6, label="CA-CFAR N$_t$=16, Swerling 1 (closed form)")
    xm = z["snr_mc_db"]
    for model, col, mk in (("sw0", C[0], "o"), ("sw1", C[1], "s")):
        y = z[f"mc_{model}"]
        ax.errorbar(xm, y, yerr=[y - z[f"mc_{model}_lo"], z[f"mc_{model}_hi"] - y], fmt=mk, mfc="none",
                    color=col, ms=5, capsize=2, lw=0, elinewidth=1,
                    label=f"Monte Carlo, {'Swerling 0' if model == 'sw0' else 'Swerling 1'} (95 % CI)")
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("single-cell SNR (dB)")
    ax.set_ylabel("probability of detection")
    ax.set_title("(a) P$_D$ vs SNR, P$_{FA}$ = 10$^{%d}$" % round(np.log10(m['pfa'])), loc="left")
    ax.legend(loc="lower right", fontsize=7.2)

    ax = axes[1]
    ax.semilogx(z["nt_sweep"], z["cfar_loss_pd05_db"], "o-", color=C[0], ms=4, label="P$_D$ = 0.5")
    ax.semilogx(z["nt_sweep"], z["cfar_loss_pd09_db"], "s--", color=C[2], ms=4, mfc="none", label="P$_D$ = 0.9")
    ax.axvline(16, color=MUTED, lw=0.8, ls=":")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xlabel("number of training cells N$_t$")
    ax.set_ylabel("CA-CFAR loss re: ideal detector (dB)")
    ax.set_title("(b) CFAR loss, Swerling 0, P$_{FA}$ = 10$^{-4}$", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig12_pd_vs_snr.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        mm = run()
        print(json.dumps({k: v for k, v in mm.items() if k != "mc_table"}, indent=1))
        for r in mm["mc_table"]:
            print(r)
    print(plot())

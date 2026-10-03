"""Part C - finite-snapshot effects on SMI/MVDR output SINR and covariance conditioning.

For the baseline scenario (N = 16, jammers 35 deg / 40 dB and -42 deg / 30 dB,
target 0 deg at 0 dB per element) the number of training snapshots K is swept
from below N to 40 N. Four estimators are compared:

1. unloaded SMI, signal-free training     -> RMB theory  (K - N + 2)/(K + 1)
2. unloaded SMI, target in training        -> rho/(1 + S(1 - rho)) theory
3. SMI with 10 dB diagonal loading, signal-free
4. SMI with 10 dB diagonal loading, target in training

The condition number of the sample covariance is recorded as a function of K
and compared with the condition number of the true covariance.

Usage: ``python experiments/snapshot_sensitivity.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab import condition_number, sample_covariance, simulate_snapshots
from array_signal_lab.io import load_results, save_results
from array_signal_lab.metrics import (
    db,
    rmb_mean_normalized_sinr,
    smi_signal_present_mean_normalized_sinr,
)
from common import JAMMERS_DEG, JNR_DB, N, SNR_DB, baseline, mean_ci_db, smi_trials

NAME = "snapshot_sensitivity"
SEED = 20261003
N_TRIALS = 600
K_VALUES = [4, 8, 12, 15, 16, 17, 18, 20, 24, 28, 32, 40, 48, 64, 96, 128, 192, 256, 384, 640]
LOADING_DB = 10.0


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    sc = baseline()
    load = 10 ** (LOADING_DB / 10)
    cases = {
        "smi_signal_free": dict(signal_in_training=False, loading=0.0),
        "smi_signal_present": dict(signal_in_training=True, loading=0.0),
        "loaded_signal_free": dict(signal_in_training=False, loading=load),
        "loaded_signal_present": dict(signal_in_training=True, loading=load),
    }
    arrays = {"K": np.array(K_VALUES, dtype=float)}
    table = []
    for name, kw in cases.items():
        mean_db, lo_db, hi_db, sing = [], [], [], []
        for k in K_VALUES:
            rho, fs, _ = smi_trials(sc, k, N_TRIALS, rng, **kw)
            m, lo, hi = mean_ci_db(rho)
            mean_db.append(m)
            lo_db.append(lo)
            hi_db.append(hi)
            sing.append(fs)
        arrays[f"{name}_mean_db"] = np.array(mean_db)
        arrays[f"{name}_lo_db"] = np.array(lo_db)
        arrays[f"{name}_hi_db"] = np.array(hi_db)
        arrays[f"{name}_singular_fraction"] = np.array(sing)
    rmb = np.array([rmb_mean_normalized_sinr(k, N) for k in K_VALUES])
    sp = np.array([smi_signal_present_mean_normalized_sinr(k, N, sc["sinr_opt"]) for k in K_VALUES])
    arrays["rmb_theory_db"] = db(rmb)
    arrays["signal_present_theory_db"] = db(sp)

    # agreement with theory (z-scores of the MC mean, using the CI half-width / 1.96 as s.e.)
    def zscores(name, theory_lin):
        m = 10 ** (arrays[f"{name}_mean_db"] / 10)
        se = (10 ** (arrays[f"{name}_hi_db"] / 10) - m) / 1.96
        return (m - theory_lin) / se

    z_rmb = zscores("smi_signal_free", rmb)
    z_sp = zscores("smi_signal_present", sp)
    for i, k in enumerate(K_VALUES):
        table.append({
            "K": k,
            "rmb_theory_db": float(db(rmb[i])) if np.isfinite(rmb[i]) else None,
            "smi_signal_free_mc_db": float(arrays["smi_signal_free_mean_db"][i]),
            "z_vs_rmb": float(z_rmb[i]) if np.isfinite(z_rmb[i]) else None,
            "signal_present_theory_db": float(db(sp[i])) if np.isfinite(sp[i]) else None,
            "smi_signal_present_mc_db": float(arrays["smi_signal_present_mean_db"][i]),
            "z_vs_signal_present_theory": float(z_sp[i]) if np.isfinite(z_sp[i]) else None,
            "loaded_signal_free_mc_db": float(arrays["loaded_signal_free_mean_db"][i]),
            "loaded_signal_present_mc_db": float(arrays["loaded_signal_present_mean_db"][i]),
            "unloaded_singular_fraction": float(arrays["smi_signal_free_singular_fraction"][i]),
        })

    # condition number of R_hat (signal-free training) vs K
    cond_med, cond_p10, cond_p90 = [], [], []
    for k in K_VALUES:
        if k < N:  # rank-deficient: condition number is infinite by construction
            cond_med.append(np.nan); cond_p10.append(np.nan); cond_p90.append(np.nan)
            continue
        c = [condition_number(sample_covariance(simulate_snapshots(sc["ula"], sc["th_j"], sc["jnr"], 1.0, k, rng)))
             for _ in range(200)]
        c = np.array(c)
        cond_med.append(np.median(c))
        cond_p10.append(np.percentile(c, 10))
        cond_p90.append(np.percentile(c, 90))
    arrays["cond_median"] = np.array(cond_med)
    arrays["cond_p10"] = np.array(cond_p10)
    arrays["cond_p90"] = np.array(cond_p90)
    cond_true = condition_number(sc["R_in"])
    # Large-dimension estimate: the N - r noise eigenvalues of R_hat spread over the
    # Marchenko-Pastur support; its lower edge is sigma^2 (1 - sqrt((N - r)/K))^2.
    # With the dominant eigenvalue estimated ~unbiasedly, cond(R_hat) ~ lambda_1 / lower edge.
    r_int = len(sc["th_j"])
    lam1 = np.linalg.eigvalsh(sc["R_in"])[-1]
    k_arr = np.array(K_VALUES, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = lam1 / (1 - np.sqrt((N - r_int) / k_arr)) ** 2
    mp[k_arr < N] = np.nan
    arrays["cond_mp_estimate"] = mp

    def first_k_within(name, thr_db=-3.0):
        m = arrays[f"{name}_mean_db"]
        ok = np.where(np.isfinite(m) & (m >= thr_db))[0]
        return int(K_VALUES[ok[0]]) if ok.size else None

    finite_rmb = np.isfinite(z_rmb)
    finite_sp = np.isfinite(z_sp)
    metrics = {
        "sinr_opt_db": float(db(sc["sinr_opt"])),
        "cond_true_Rin": cond_true,
        "table": table,
        "max_abs_z_vs_rmb": float(np.max(np.abs(z_rmb[finite_rmb]))),
        "max_abs_z_vs_signal_present_theory": float(np.max(np.abs(z_sp[finite_sp]))),
        "rmb_K_for_3dB_loss_theory": "K = 2N - 3 = %d" % (2 * N - 3),
        "smallest_K_in_grid_with_mean_loss_le_3dB": {
            name: first_k_within(name) for name in cases
        },
    }
    config = {
        "N": N, "jammers_deg": JAMMERS_DEG, "jnr_db": JNR_DB, "snr_db_per_element": SNR_DB,
        "K_values": K_VALUES, "n_trials_per_point": N_TRIALS, "loading_db_rel_noise": LOADING_DB,
        "cond_trials_per_K": 200, "seed": SEED,
    }
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    K = z["K"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]
    styles = [
        ("smi_signal_free", "SMI, signal-free training", C[0], "o"),
        ("smi_signal_present", "SMI, target in training", C[1], "s"),
        ("loaded_signal_free", "SMI + 10 dB loading, signal-free", C[2], "^"),
        ("loaded_signal_present", "SMI + 10 dB loading, target in training", C[6], "v"),
    ]
    for key, lab, col, mk in styles:
        m = z[f"{key}_mean_db"]
        ok = np.isfinite(m)
        ax.errorbar(K[ok], m[ok], yerr=[m[ok] - z[f"{key}_lo_db"][ok], z[f"{key}_hi_db"][ok] - m[ok]],
                    fmt=mk, ms=4.5, mfc="none", color=col, capsize=2, lw=0, elinewidth=1, label=lab)
    kk = np.arange(16, 641)
    from array_signal_lab.metrics import rmb_mean_normalized_sinr, smi_signal_present_mean_normalized_sinr
    ax.plot(kk, [10 * np.log10(rmb_mean_normalized_sinr(k, 16)) for k in kk], color=C[0], lw=1.2,
            label="RMB theory (K − N + 2)/(K + 1)")
    s = 10 ** (rec["metrics"]["sinr_opt_db"] / 10)
    kk2 = np.unique(np.geomspace(16, 640, 60).astype(int))
    ax.plot(kk2, [10 * np.log10(smi_signal_present_mean_normalized_sinr(k, 16, s)) for k in kk2],
            color=C[1], lw=1.2, ls="--", label=r"theory, target in training: $E[\rho/(1+S(1-\rho))]$")
    ax.axvline(16, color=MUTED, lw=0.8, ls=":")
    ax.text(15, -21.5, "K = N", ha="right", fontsize=7.5, color=MUTED)
    ax.axhline(-3, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.set_xlim(3, 900)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 640])
    ax.set_ylim(-22, 0.5)
    ax.set_xlabel("training snapshots K")
    ax.set_ylabel(r"mean $\rho$ = SINR / SINR$_{\rm opt}$ (dB)")
    ax.set_title("(a) Normalized output SINR vs snapshots (N = 16)", loc="left")
    ax.legend(loc="lower right", fontsize=7.2)
    ax.text(4.2, -11.5, "unloaded SMI undefined\nfor K < N (rank K)", fontsize=7.5, color=MUTED, va="top")

    ax = axes[1]
    c = z["cond_median"]
    ok = np.isfinite(c)
    ax.loglog(K[ok], c[ok], "o-", color=C[0], ms=4, label=r"median cond($\hat R$), signal-free")
    ax.fill_between(K[ok], z["cond_p10"][ok], z["cond_p90"][ok], color=C[0], alpha=0.18, lw=0,
                    label="10–90 % range")
    ax.loglog(K[ok], z["cond_mp_estimate"][ok], ls="--", color=C[2], lw=1.2,
              label=r"$\lambda_1(R_{in})/(1-\sqrt{(N-r)/K})^2$ (Marchenko–Pastur edge)")
    ax.axhline(rec["metrics"]["cond_true_Rin"], color=C[1], ls="--", lw=1.2,
               label=f"cond($R_{{in}}$) = {rec['metrics']['cond_true_Rin']:.3g}")
    ax.axvline(16, color=MUTED, lw=0.8, ls=":")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.set_xlim(3, 900)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 640])
    ax.set_xlabel("training snapshots K")
    ax.set_ylabel("2-norm condition number")
    ax.set_title(r"(b) Conditioning of $\hat R$ (K < N: singular, not shown)", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig03_sinr_vs_snapshots.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        m = run()
        print(json.dumps({k: v for k, v in m.items() if k != "table"}, indent=1))
        for r in m["table"]:
            print(r)
    print(plot())

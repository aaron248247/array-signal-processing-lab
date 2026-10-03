"""Part C - single-jammer sweeps over jammer angle and jammer-to-noise ratio.

Single jammer in white noise, N = 16, target at broadside, SNR 0 dB/element.
Closed forms (matrix inversion lemma, beta = a_s^H a_j):

    SINR_opt  = snr (N - jnr |beta|^2 / (1 + N jnr))
    SINR_conv = snr N^2 / (N + jnr |beta|^2)

are compared with Monte Carlo SMI (K = 2N, signal-free training) with and
without 10 dB diagonal loading. The beam-pointing error of the clairvoyant
MVDR pattern (location of the main-lobe maximum relative to the look
direction) is recorded as the jammer approaches the main lobe.

Usage: ``python experiments/jammer_sweep.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize_scalar

from array_signal_lab import (
    ULA,
    condition_number,
    mvdr_weights,
    response,
    steering_vector,
    theoretical_covariance,
)
from array_signal_lab.io import load_results, save_results
from array_signal_lab.metrics import (
    conventional_sinr_single_jammer,
    db,
    optimal_sinr_single_jammer,
)
from common import mean_ci_db, smi_trials

NAME = "jammer_sweep"
SEED = 20261006
N = 16
SNR_DB = 0.0
JNR_ANGLE_SWEEP_DB = 30.0
JAMMER_ANGLE_FIXED_DEG = 35.0
ANGLES_FINE = np.linspace(-90, 90, 721)
ANGLES_MC = np.arange(-88.0, 89.0, 4.0) + 1.0  # offset so 0 deg (target) is not sampled
JNR_SWEEP_DB = np.arange(-10.0, 60.1, 5.0)
K_SMI = 2 * N
N_TRIALS = 200


def _scenario(th_j, jnr):
    ula = ULA(N)
    snr = 10 ** (SNR_DB / 10)
    R_in = theoretical_covariance(ula, [th_j], [jnr], 1.0)
    a_s = steering_vector(ula, 0.0)
    return dict(ula=ula, th_s=0.0, th_j=np.array([th_j]), jnr=np.array([jnr]), snr=snr, a_s=a_s, R_in=R_in)


POINTING_WINDOW_DEG = 7.0  # just inside the conventional first nulls (+-7.18 deg)


def _pointing_error_deg(ula, w):
    """Location of the maximum of |w^H a| within +-POINTING_WINDOW_DEG of the look direction.

    Returns NaN when the maximum lies on the window edge, i.e. when the
    pattern has no interior main-lobe peak (jammer inside the main lobe).
    """
    grid = np.deg2rad(np.linspace(-POINTING_WINDOW_DEG, POINTING_WINDOW_DEG, 1401))
    mag = np.abs(response(w, ula, grid))
    i = int(np.argmax(mag))
    if i in (0, grid.size - 1):
        return float("nan")
    f = lambda t: -np.abs(response(w, ula, t))  # noqa: E731
    r = minimize_scalar(f, bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-12})
    return float(np.rad2deg(r.x))


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    ula = ULA(N)
    a_s = steering_vector(ula, 0.0)
    snr = 10 ** (SNR_DB / 10)
    jnr = 10 ** (JNR_ANGLE_SWEEP_DB / 10)
    arrays = {"angles_fine_deg": ANGLES_FINE, "angles_mc_deg": ANGLES_MC, "jnr_sweep_db": JNR_SWEEP_DB}

    # (a) angle sweep, closed forms
    opt, conv, perr = [], [], []
    for ang in ANGLES_FINE:
        a_j = steering_vector(ula, np.deg2rad(ang))
        beta = np.vdot(a_s, a_j)
        opt.append(db(optimal_sinr_single_jammer(N, snr, jnr, beta)))
        conv.append(db(conventional_sinr_single_jammer(N, snr, jnr, beta)))
        sc = _scenario(np.deg2rad(ang), jnr)
        w = mvdr_weights(sc["R_in"], a_s)
        perr.append(_pointing_error_deg(ula, w))
    arrays["angle_opt_db"] = np.array(opt)
    arrays["angle_conv_db"] = np.array(conv)
    arrays["angle_pointing_error_deg"] = np.array(perr)
    # MC
    for name, lnr in (("smi", 0.0), ("smi_load10", 10.0)):
        vals = []
        for ang in ANGLES_MC:
            sc = _scenario(np.deg2rad(ang), jnr)
            sopt = optimal_sinr_single_jammer(N, snr, jnr, np.vdot(a_s, steering_vector(ula, np.deg2rad(ang))))
            rho, _, _ = smi_trials(sc, K_SMI, N_TRIALS, rng, loading=lnr and 10 ** (lnr / 10))
            vals.append(mean_ci_db(rho)[0] + db(sopt))
        arrays[f"angle_{name}_db"] = np.array(vals)

    # (c) JNR sweep at fixed angle
    th_j = np.deg2rad(JAMMER_ANGLE_FIXED_DEG)
    beta = np.vdot(a_s, steering_vector(ula, th_j))
    arrays["jnr_opt_db"] = np.array([db(optimal_sinr_single_jammer(N, snr, 10 ** (j / 10), beta)) for j in JNR_SWEEP_DB])
    arrays["jnr_conv_db"] = np.array([db(conventional_sinr_single_jammer(N, snr, 10 ** (j / 10), beta)) for j in JNR_SWEEP_DB])
    conds = []
    for name, lnr in (("smi", 0.0), ("smi_load10", 10.0)):
        vals = []
        for j in JNR_SWEEP_DB:
            sc = _scenario(th_j, 10 ** (j / 10))
            sopt = optimal_sinr_single_jammer(N, snr, 10 ** (j / 10), beta)
            rho, _, _ = smi_trials(sc, K_SMI, N_TRIALS, rng, loading=lnr and 10 ** (lnr / 10))
            vals.append(mean_ci_db(rho)[0] + db(sopt))
            if name == "smi":
                conds.append(condition_number(sc["R_in"]))
        arrays[f"jnr_{name}_db"] = np.array(vals)
    arrays["jnr_cond_Rin"] = np.array(conds)

    pe = np.abs(arrays["angle_pointing_error_deg"])
    undefined = ANGLES_FINE[~np.isfinite(pe)]
    fn_half = np.rad2deg(np.arcsin(2.0 / N))  # first null of the conventional beam, 7.18 deg

    def at(arr, grid, v):
        return float(arr[int(np.argmin(np.abs(grid - v)))])

    metrics = {
        "no_jammer_array_snr_db": float(db(N * snr)),
        "angle_sweep_jnr_db": JNR_ANGLE_SWEEP_DB,
        "first_null_half_width_deg": float(fn_half),
        "opt_sinr_db_at_jammer_deg": {f"{a:+.0f}": at(arrays["angle_opt_db"], ANGLES_FINE, a)
                                       for a in (2.0, 4.0, 7.18, 10.0, 35.0)},
        "conv_sinr_db_at_jammer_deg": {f"{a:+.0f}": at(arrays["angle_conv_db"], ANGLES_FINE, a)
                                        for a in (2.0, 4.0, 7.18, 10.0, 35.0)},
        "max_abs_pointing_error_deg_outside_mainlobe": float(pe[np.abs(ANGLES_FINE) > fn_half].max()),
        "max_abs_pointing_error_deg_where_defined": float(np.nanmax(pe)),
        "jammer_angle_of_max_pointing_error_deg": float(ANGLES_FINE[int(np.nanargmax(pe))]),
        "pointing_undefined_jammer_angles_deg": [float(undefined.min()), float(undefined.max())] if undefined.size else None,
        "pointing_window_deg": POINTING_WINDOW_DEG,
        "jnr_sweep_jammer_deg": JAMMER_ANGLE_FIXED_DEG,
        "jnr_sweep": [
            {"jnr_db": float(j), "opt_db": float(o), "conv_db": float(c), "smi_db": float(s),
             "smi_load10_db": float(l), "cond_Rin": float(k)}
            for j, o, c, s, l, k in zip(JNR_SWEEP_DB, arrays["jnr_opt_db"], arrays["jnr_conv_db"],
                                        arrays["jnr_smi_db"], arrays["jnr_smi_load10_db"], arrays["jnr_cond_Rin"])
        ],
    }
    config = {
        "N": N, "snr_db_per_element": SNR_DB, "jnr_angle_sweep_db": JNR_ANGLE_SWEEP_DB,
        "jammer_angle_fixed_deg": JAMMER_ANGLE_FIXED_DEG, "K_smi": K_SMI, "n_trials_per_point": N_TRIALS,
        "angles_mc_deg": ANGLES_MC, "jnr_sweep_db": JNR_SWEEP_DB, "seed": SEED,
    }
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    m = rec["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.plot(z["angles_fine_deg"], z["angle_opt_db"], color=C[1], label="optimum (closed form)")
    ax.plot(z["angles_fine_deg"], z["angle_conv_db"], color=C[0], label="conventional (closed form)")
    ax.plot(z["angles_mc_deg"], z["angle_smi_db"], "o", ms=4, mfc="none", color=C[2], label="SMI, K = 2N (MC mean)")
    ax.plot(z["angles_mc_deg"], z["angle_smi_load10_db"], "^", ms=4, mfc="none", color=C[6],
            label="SMI, K = 2N, 10 dB loading")
    ax.axhline(m["no_jammer_array_snr_db"], color=MUTED, ls=":", lw=0.9)
    ax.text(-88, m["no_jammer_array_snr_db"] + 0.7, "no jammer: N·SNR", fontsize=7.5, color=MUTED)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-25, 15)
    ax.set_xlabel("jammer angle (deg); target at 0°")
    ax.set_ylabel("output SINR (dB)")
    ax.set_title(f"(a) SINR vs jammer angle, JNR = {m['angle_sweep_jnr_db']:.0f} dB", loc="left")
    ax.legend(loc="lower left", fontsize=7.2)

    ax = axes[1]
    ax.plot(z["angles_fine_deg"], z["angle_pointing_error_deg"], color=C[1])
    fh = m["first_null_half_width_deg"]
    ax.axvspan(-fh, fh, color=C[0], alpha=0.10, lw=0)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("jammer angle (deg)")
    ax.set_ylabel("MVDR main-lobe peak − look direction (deg)")
    ax.set_title("(b) MVDR beam-pointing error (true $R_{in}$)", loc="left")
    u = m["pointing_undefined_jammer_angles_deg"]
    if u:
        ax.axvspan(u[0], u[1], color=MUTED, alpha=0.25, lw=0)
        ax.text(u[1] + 0.5, -2.6, "no interior peak\nwithin ±7°", fontsize=7.5, color=MUTED)
    ax.text(0.02, 0.04, "shaded: conventional main lobe (first nulls ±7.18°)", transform=ax.transAxes,
            fontsize=7.5, color=MUTED)

    ax = axes[2]
    J = z["jnr_sweep_db"]
    ax.plot(J, z["jnr_opt_db"], color=C[1], label="optimum (closed form)")
    ax.plot(J, z["jnr_conv_db"], color=C[0], label="conventional (closed form)")
    ax.plot(J, z["jnr_smi_db"], "o", ms=4, mfc="none", color=C[2], label="SMI, K = 2N")
    ax.plot(J, z["jnr_smi_load10_db"], "^", ms=4, mfc="none", color=C[6], label="SMI, K = 2N, 10 dB loading")
    ax.set_xlabel("JNR per element (dB)")
    ax.set_ylabel("output SINR (dB)")
    ax.set_title(f"(c) SINR vs JNR, jammer at {rec['config']['jammer_angle_fixed_deg']:.0f}°", loc="left")
    ax.legend(loc="lower left", fontsize=7.2)
    fig.tight_layout()
    return save(fig, "fig06_jammer_angle_jnr.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        m = run()
        print(json.dumps({k: v for k, v in m.items() if k != "jnr_sweep"}, indent=1))
        for r in m["jnr_sweep"]:
            print(r)
    print(plot())

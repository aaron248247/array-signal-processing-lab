"""Part A - ULA model verification against the closed-form array factor.

Questions
---------
1. Does the numerical response ``w^H a(theta)`` reproduce the closed-form
   array factor to machine precision for broadside and steered beams?
2. Is the main-lobe peak at the commanded angle?
3. Are the nulls where ``sin(theta) = sin(theta0) + m lambda / (N d)`` predicts?
4. Do half-power beamwidth and peak sidelobe level follow known limits?
5. Does an independent estimator (beam-scanning simulated plane-wave snapshots)
   agree with ``|AF|^2``?

Usage: ``python experiments/ula_verification.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from array_signal_lab import (
    ULA,
    angle_scan,
    array_factor_analytical,
    conventional_weights,
    first_null_beamwidth,
    null_angles,
    response,
    steering_vector,
)
from array_signal_lab.io import load_results, save_results

NAME = "ula_verification"
SEED = 20261001


def _refine_peak(ula, w, theta_guess, half_width):
    f = lambda t: -np.abs(response(w, ula, t))  # noqa: E731
    r = minimize_scalar(f, bounds=(theta_guess - half_width, theta_guess + half_width),
                        method="bounded", options={"xatol": 1e-12})
    return r.x


def _refine_null(ula, w, th0, theta_guess, half_width):
    """Locate a null of the *numerical* response by bracketing a sign change.

    The numerical response is multiplied by exp(-j (N-1) psi / 2) to remove the
    linear phase (element-0 reference); the remainder is real and changes sign
    through each null, so Brent's method converges to the zero.
    """
    n = ula.n_elements

    def amp(t):
        af = response(w, ula, t)
        psi = ula.wavenumber * ula.spacing * (np.sin(t) - np.sin(th0))
        return np.real(af * np.exp(-1j * (n - 1) * psi / 2))

    a, b = theta_guess - half_width, theta_guess + half_width
    if np.sign(amp(a)) == np.sign(amp(b)):
        return np.nan
    return brentq(amp, a, b, xtol=1e-14)


def _hpbw(ula, w, theta_peak):
    """Half-power beamwidth from root-finding |AF|^2 = 1/2 on both sides of the peak."""
    g = lambda t: np.abs(response(w, ula, t)) ** 2 - 0.5  # noqa: E731
    fn = first_null_beamwidth(ula, theta_peak) / 2
    lo = brentq(g, theta_peak - 0.999 * fn, theta_peak)
    hi = brentq(g, theta_peak, theta_peak + 0.999 * fn)
    return hi - lo


def run():
    t0 = time.perf_counter()
    thetas = np.deg2rad(np.linspace(-90.0, 90.0, 180001))  # 0.001 deg grid
    cases = [(16, 0.0), (16, 30.0), (16, -45.0), (8, 0.0), (8, 20.0)]
    rows = []
    arrays = {"theta_sub_deg": np.rad2deg(thetas[::10]), "theta_markers_deg": np.rad2deg(thetas[::5000])}
    for n, th0_deg in cases:
        ula = ULA(n_elements=n, spacing_wavelengths=0.5)
        th0 = np.deg2rad(th0_deg)
        w = conventional_weights(ula, th0)
        num = response(w, ula, thetas)
        ana = array_factor_analytical(ula, thetas, th0)
        err = np.abs(num - ana)
        # main lobe
        grid_peak = thetas[np.argmax(np.abs(num))]
        fine_peak = _refine_peak(ula, w, grid_peak, np.deg2rad(0.01))
        # nulls
        nulls_an = null_angles(ula, th0)
        nulls_an_inner = nulls_an[np.abs(np.sin(nulls_an)) < 1 - 1e-9]  # exclude endfire zeros
        hw = 0.25 / (n * 0.5)  # quarter of the null spacing in sin-space, conservative
        nulls_num = np.array([_refine_null(ula, w, th0, t, hw * 0.5) for t in nulls_an_inner])
        null_err = np.abs(nulls_num - nulls_an_inner)
        # sidelobes: largest local maximum outside the first nulls
        mag_db = 20 * np.log10(np.maximum(np.abs(num), 1e-300))
        fn_lo = np.arcsin(np.sin(th0) - 1.0 / (n * 0.5))
        fn_hi = np.arcsin(min(1.0, np.sin(th0) + 1.0 / (n * 0.5)))
        outside = (thetas < fn_lo) | (thetas > fn_hi)
        psl_db = mag_db[outside].max()
        hpbw = _hpbw(ula, w, fine_peak)
        hpbw_approx = 0.886 / (n * 0.5 * np.cos(th0))  # standard approximation
        key = f"N{n}_th{th0_deg:+.0f}"
        # Stored for plotting (the full 0.001-degree complex arrays are ~30 MB and not needed):
        # |AF| in dB on a 0.01-degree subgrid, closed form at 5-degree markers, and the
        # maximum error in each 1-degree bin. All metrics above use the full grid.
        arrays[f"num_db_{key}"] = (20 * np.log10(np.maximum(np.abs(num[::10]), 1e-12))).astype(np.float32)
        arrays[f"ana_db_markers_{key}"] = 20 * np.log10(np.maximum(np.abs(ana[::5000]), 1e-12))
        arrays[f"err_binmax_{key}"] = err[:-1].reshape(180, 1000).max(axis=1)
        arrays[f"nulls_an_{key}"] = np.rad2deg(nulls_an_inner)
        arrays[f"nulls_num_{key}"] = np.rad2deg(nulls_num)
        rows.append({
            "case": key,
            "N": n,
            "theta0_deg": th0_deg,
            "max_abs_af_error": float(err.max()),
            "peak_grid_deg": float(np.rad2deg(grid_peak)),
            "peak_refined_deg": float(np.rad2deg(fine_peak)),
            "peak_error_deg": float(abs(np.rad2deg(fine_peak) - th0_deg)),
            "n_interior_nulls": int(nulls_an_inner.size),
            "max_null_location_error_deg": float(np.rad2deg(np.nanmax(null_err))),
            "max_abs_af_at_analytical_nulls": float(np.abs(response(w, ula, nulls_an)).max()),
            "peak_sidelobe_db": float(psl_db),
            "hpbw_deg": float(np.rad2deg(hpbw)),
            "hpbw_approx_0p886_deg": float(np.rad2deg(hpbw_approx)),
            "fnbw_deg": float(np.rad2deg(first_null_beamwidth(ula, th0))),
        })

    # Independent estimator: beam-scan of simulated plane-wave snapshots vs |AF|^2
    rng = np.random.default_rng(SEED)
    ula = ULA(16)
    th_s = np.deg2rad(30.0)
    k_snap = 4096
    s = (rng.standard_normal(k_snap) + 1j * rng.standard_normal(k_snap)) / np.sqrt(2)
    X = np.outer(steering_vector(ula, th_s), s)
    scan = np.deg2rad(np.linspace(-90, 90, 3601))
    p_scan = angle_scan(ula, X, scan)
    p_expected = np.mean(np.abs(s) ** 2) * np.abs(array_factor_analytical(ula, th_s, scan)) ** 2
    scan_rel_err = float(np.max(np.abs(p_scan - p_expected)) / p_expected.max())
    arrays["scan_deg"] = np.rad2deg(scan)
    arrays["scan_power"] = p_scan
    arrays["scan_expected"] = p_expected

    # Grating-lobe limiting case: d = lambda, broadside -> lobes at +-90 deg; steered 20 deg ->
    # grating lobe at sin = sin(20) - 1
    ula_g = ULA(16, 1.0)
    th0 = np.deg2rad(20.0)
    w = conventional_weights(ula_g, th0)
    gl_pred = np.arcsin(np.sin(th0) - 1.0)
    gl_resp = float(np.abs(response(w, ula_g, gl_pred)))

    # large-N limit: AF -> sinc; first sidelobe at the root of tan(x) = x near 1.43 pi
    x1 = brentq(lambda x: np.tan(x) - x, 1.01 * np.pi, 1.49 * np.pi)
    psl_limit = 20 * np.log10(abs(np.sin(x1) / x1))

    metrics = {
        "cases": rows,
        "beam_scan_vs_af2_max_rel_error": scan_rel_err,
        "grating_lobe_d1_theta20_pred_deg": float(np.rad2deg(gl_pred)),
        "grating_lobe_d1_theta20_abs_response": gl_resp,
        "psl_large_N_limit_db": float(psl_limit),
    }
    config = {
        "angle_grid_deg": [-90.0, 90.0, 180001],
        "cases_N_theta0deg": cases,
        "spacing_wavelengths": 0.5,
        "beam_scan_snapshots": k_snap,
        "beam_scan_source_deg": 30.0,
        "seed": SEED,
    }
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    from plotstyle import C, MUTED, panel_label, save, setup
    import matplotlib.pyplot as plt

    setup()
    rec, z = load_results(NAME)
    th = z["theta_sub_deg"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    for i, (key, lab) in enumerate([("N16_th+0", r"$\theta_0=0^\circ$"), ("N16_th+30", r"$\theta_0=30^\circ$")]):
        mag = z[f"num_db_{key}"].astype(float)
        ax.plot(th, mag, color=C[i], label=f"numerical, {lab}", lw=1.4)
        ax.plot(z["theta_markers_deg"], z[f"ana_db_markers_{key}"], ls="none", marker="o", ms=4, mfc="none", color=C[i],
                label=f"closed form, {lab}")
        nulls = z[f"nulls_an_{key}"]
        ax.plot(nulls, np.full_like(nulls, -60.0 + 3.0 * i), ls="none", marker="^", ms=5,
                mec="white", mew=0.6, color=C[i], zorder=5)
    ax.set_ylim(-65, 3)
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r"angle from broadside $\theta$ (deg)")
    ax.set_ylabel(r"$20\log_{10}|w^H a(\theta)|$ (dB)")
    ax.set_title("(a) Array factor, N = 16, d = λ/2", loc="left")
    ax.legend(loc="upper left", ncol=1, fontsize=7.5)
    ax.text(0.99, 0.02, "triangles: analytical nulls", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7.5, color=MUTED)

    ax = axes[1]
    for i, key in enumerate(["N16_th+0", "N16_th+30", "N16_th-45", "N8_th+0"]):
        # maximum error in 1-degree bins (the raw 0.001-degree trace is unreadable)
        e = z[f"err_binmax_{key}"]
        centers = np.arange(-89.5, 90, 1.0)
        ax.semilogy(centers, np.maximum(e, 1e-18), color=C[i], lw=1.3,
                    ls=["-", "--", "-.", ":"][i], label=key.replace("_th", r", $\theta_0$=") + "°")
    ax.set_xlim(-90, 90)
    ax.set_ylim(1e-17, 1e-13)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"max $|AF_{\rm num}-AF_{\rm closed}|$ per 1° bin")
    ax.set_title("(b) Numerical vs closed-form error", loc="left")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)

    ax = axes[2]
    markers = ["o", "s", "^", "D", "v"]
    for i, row in enumerate(rec["metrics"]["cases"]):
        key = row["case"]
        an = z[f"nulls_an_{key}"]
        nu = z[f"nulls_num_{key}"]
        err = np.maximum(np.abs(nu - an), 1e-16)  # exact zeros drawn at the 1e-16 floor
        ax.semilogy(an, err, ls="none", marker=markers[i], ms=5, mfc="none", color=C[i],
                    label=key.replace("_th", r", $\theta_0$=") + "°")
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r"analytical null location (deg)")
    ax.set_ylabel("|refined numerical null − analytical| (deg)")
    ax.set_title("(c) Null-location error", loc="left")
    ax.set_ylim(5e-17, 1e-12)
    ax.legend(loc="upper center", fontsize=7.5, ncol=3, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    return save(fig, "fig01_ula_verification.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        m = run()
        for r in m["cases"]:
            print(r)
        print({k: v for k, v in m.items() if k != "cases"})
    print(plot())

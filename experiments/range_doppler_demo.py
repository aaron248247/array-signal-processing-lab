"""Part D - LFM pulse compression, range/Doppler recovery and FFT-binning effects.

Generic parameters (see RadarParams defaults): f_c = 3 GHz, B = 5 MHz,
T = 10 us (BT = 50), f_s = 10 MHz (2x oversampled), PRF = 10 kHz, M = 64.

Questions
---------
1. Does the sampled matched-filter response match the closed-form LFM
   autocorrelation, and is the measured mainlobe width ~1/B?
2. Are two equal targets resolved at the nominal c/(2B) separation?
3. Is the post-compression SNR gain equal to the pulse length in samples?
4. Is the Doppler-induced LFM delay shift equal to -f_D T / B?
5. Are injected ranges and velocities recovered in a noisy range-Doppler map?
6. How large are the grid-quantization (binning) errors and straddle/scalloping
   losses, and how much does interpolation or zero padding recover?

Produces figures 7, 8 and 9.

Usage: ``python experiments/range_doppler_demo.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import brentq

from array_signal_lab.doppler import doppler_axis, doppler_process, scalloping_loss_db, window
from array_signal_lab.matched_filter import estimate_delay_samples, matched_filter, parabolic_peak
from array_signal_lab.io import load_results, save_results
from array_signal_lab.radar_signal import PointTarget, RadarParams, simulate_cpi
from array_signal_lab.waveforms import (
    lfm,
    lfm_autocorrelation_magnitude,
    lfm_coupling_delay,
    lfm_pulse,
)

NAME = "range_doppler_demo"
SEED = 20261007
P = RadarParams()
TARGETS = [
    PointTarget(4321.7, 101.3, -15.0, 0.3),
    PointTarget(6200.0, -45.6, -20.0, 1.1),
    PointTarget(9007.4, 12.2, -25.0, 2.0),
]


def _mf_tests(rng):
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    L = h.size
    out, arrays = {}, {}

    # (1) noiseless MF vs closed form at a fractional delay
    r0 = TARGETS[0].range_m
    X = simulate_cpi(P, [PointTarget(r0, 0.0, 0.0)], include_range_migration=False)
    y = matched_filter(X[0], h)
    lag = np.arange(P.n_fast)
    tau0 = 2 * r0 / P.c
    ref = lfm_autocorrelation_magnitude(lag / P.fs_hz - tau0, P.bandwidth_hz, P.pulse_s)
    yn = np.abs(y) / np.sqrt(L)
    win = slice(int(tau0 * P.fs_hz) - 30, int(tau0 * P.fs_hz) + 31)
    arrays["mf_lag"] = lag[win]
    arrays["mf_mag"] = yn[win]
    tt = np.linspace(lag[win][0], lag[win][-1], 2001)
    arrays["mf_ref_lag"] = tt
    arrays["mf_ref"] = lfm_autocorrelation_magnitude(tt / P.fs_hz - tau0, P.bandwidth_hz, P.pulse_s)
    out["mf_vs_closed_form_max_abs_dev_near_peak"] = float(np.max(np.abs(yn[win] - ref[win])))
    out["mf_vs_closed_form_peak_dev"] = float(abs(yn[win].max() - ref[win].max()))

    # mainlobe width from a heavily oversampled MF (continuous-time proxy)
    fs_hi = 200e6
    h_hi = lfm_pulse(P.bandwidth_hz, P.pulse_s, fs_hi)
    xh = np.concatenate([np.zeros(h_hi.size), h_hi, np.zeros(h_hi.size)])
    yh = np.abs(matched_filter(xh, h_hi)) / np.sqrt(h_hi.size)
    tl = (np.arange(xh.size) - h_hi.size) / fs_hi
    pk = int(np.argmax(yh))
    right = yh[pk:]
    i3 = int(np.argmax(right < 10 ** (-3 / 20)))
    i4 = int(np.argmax(right < 10 ** (-4 / 20)))
    inull = int(np.argmax(np.diff(right) > 0))  # first local minimum
    out["mainlobe_3dB_width_x_B"] = float(2 * tl[pk + i3] * P.bandwidth_hz)
    out["mainlobe_4dB_width_x_B"] = float(2 * tl[pk + i4] * P.bandwidth_hz)
    out["first_null_delay_x_B"] = float(tl[pk + inull] * P.bandwidth_hz)
    # closed-form first null: B tau (1 - tau/T) = 1
    tn = brentq(lambda t: P.bandwidth_hz * t * (1 - t / P.pulse_s) - 1, 1e-9, P.pulse_s / 2)
    out["first_null_delay_x_B_closed_form"] = float(tn * P.bandwidth_hz)

    # (2) two-target resolution. Carrier phase makes the relative phase of two echoes
    # arbitrary, so the result is reported for relative phases 0, pi/2 and pi. A 100 MHz
    # oversampled matched filter is used as a continuous-time proxy.
    fs_r = 100e6
    h_r = lfm_pulse(P.bandwidth_hz, P.pulse_s, fs_r)
    t_r = np.arange(4 * h_r.size) / fs_r
    t0_r = h_r.size / fs_r
    res = {}
    arrays["res_tau_x_B"] = (t_r - t0_r) * P.bandwidth_hz
    for frac in (0.75, 1.0, 1.25, 1.5, 2.0):
        dtau = frac / P.bandwidth_hz  # separation in delay; Delta R = frac * c / (2B)
        for ph_name, ph in (("0", 0.0), ("pi/2", np.pi / 2), ("pi", np.pi)):
            x = lfm(t_r - t0_r, P.bandwidth_hz, P.pulse_s) + np.exp(1j * ph) * lfm(t_r - t0_r - dtau, P.bandwidth_hz, P.pulse_s)
            ya = np.abs(matched_filter(x, h_r)) / np.sqrt(h_r.size)
            i_a, i_b = int(round(t0_r * fs_r)), int(round((t0_r + dtau) * fs_r))
            region = ya[i_a - 20:i_b + 21]
            mx = [i for i in range(1, region.size - 1) if region[i] >= region[i - 1] and region[i] > region[i + 1]
                  and region[i] > 0.3 * region.max()]
            if len(mx) >= 2:
                i1, i2 = mx[0], mx[-1]
                dip_db = float(20 * np.log10(region[i1:i2 + 1].min() / min(region[i1], region[i2])))
            else:
                dip_db = 0.0
            res[f"dR={frac:.2f}c/2B,phase={ph_name}"] = {"n_peaks": len(mx), "dip_db": dip_db}
            if frac == 1.0:
                arrays[f"res_1.0_phase_{ph_name.replace('/', '_')}"] = ya
    out["two_target_resolution"] = res

    # (3) SNR gain: on-bin target, many independent pulses
    snr_db = -10.0
    r_on = 250 * P.range_bin
    n_trials = 4000
    peaks = []
    noise_cells = []
    clean = simulate_cpi(P, [PointTarget(r_on, 0.0, snr_db)], include_range_migration=False)[0]
    for _ in range(n_trials // 50):
        Xn = clean[None, :] + (rng.standard_normal((50, P.n_fast)) + 1j * rng.standard_normal((50, P.n_fast))) / np.sqrt(2)
        Y = matched_filter(Xn, h, axis=1)
        peaks.append(np.abs(Y[:, 250]) ** 2)
        noise_cells.append(np.abs(Y[:, 500:800]) ** 2)
    peaks = np.concatenate(peaks)
    noise = np.concatenate(noise_cells).ravel()
    snr_out = (peaks.mean() - noise.mean()) / noise.mean()
    # standard error by delta method on the mean of peaks
    se = peaks.std(ddof=1) / np.sqrt(peaks.size) / noise.mean()
    out["snr_gain"] = {
        "input_snr_db": snr_db,
        "pulse_samples_L": L,
        "expected_output_snr_db": float(snr_db + 10 * np.log10(L)),
        "measured_output_snr_db": float(10 * np.log10(snr_out)),
        "measured_output_snr_linear": float(snr_out),
        "measured_linear_stderr": float(se),
        "expected_output_snr_linear": float(10 ** (snr_db / 10) * L),
        "noise_variance_after_mf": float(noise.mean()),
        "n_trials": n_trials,
    }

    # (4) range-Doppler coupling
    fds = np.linspace(-500e3, 500e3, 41)
    meas = []
    n = np.arange(P.n_fast)
    for fd in fds:
        x = np.zeros(P.n_fast, dtype=complex)
        x[300:300 + L] = h
        x = x * np.exp(2j * np.pi * fd * n / P.fs_hz)
        meas.append(estimate_delay_samples(matched_filter(x, h)) - 300)
    meas = np.array(meas)
    pred = np.array([lfm_coupling_delay(fd, P.bandwidth_hz, P.pulse_s) * P.fs_hz for fd in fds])
    arrays["coupling_fd_hz"] = fds
    arrays["coupling_meas_samples"] = meas
    arrays["coupling_pred_samples"] = pred
    out["coupling_max_abs_dev_samples"] = float(np.max(np.abs(meas - pred)))
    fd_max = P.prf_hz / 2
    out["coupling_range_bias_at_max_unambiguous_doppler_m"] = float(
        -lfm_coupling_delay(fd_max, P.bandwidth_hz, P.pulse_s) * P.c / 2)
    return out, arrays


def _rd_map(rng):
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    X = simulate_cpi(P, TARGETS, 1.0, rng)
    Y = matched_filter(X, h, axis=1)
    out, arrays = {}, {}
    est = []
    for win in ("rect", "hann"):
        RD = doppler_process(Y, win)
        fd_axis = doppler_axis(P.prf_hz, P.n_pulses)
        P_db = 20 * np.log10(np.abs(RD) + 1e-12)
        arrays[f"rd_{win}_db"] = P_db.astype(np.float32)
        for i, tg in enumerate(TARGETS):
            # local search window around truth: +-3 range bins, +-3 Doppler bins
            l0 = int(round(2 * tg.range_m / P.c * P.fs_hz))
            k0 = int(np.argmin(np.abs(fd_axis - P.doppler_hz(tg.velocity_mps))))
            sub = np.abs(RD[k0 - 3:k0 + 4, l0 - 3:l0 + 4])
            dk, dl = np.unravel_index(np.argmax(sub), sub.shape)
            k, l = k0 - 3 + dk, l0 - 3 + dl
            r_grid = l * P.range_bin
            v_grid = fd_axis[k] * P.wavelength / 2
            r_int = (l + parabolic_peak(np.abs(RD[k, :]), l)) * P.range_bin
            v_int = (fd_axis[k] + parabolic_peak(np.abs(RD[:, l]), k) * P.doppler_bin) * P.wavelength / 2
            noise_floor = np.median(np.abs(RD) ** 2) / np.log(2)  # median of exp(1) = ln 2
            est.append({
                "window": win, "target": i, "true_range_m": tg.range_m, "true_velocity_mps": tg.velocity_mps,
                "range_grid_m": float(r_grid), "range_interp_m": float(r_int),
                "velocity_grid_mps": float(v_grid), "velocity_interp_mps": float(v_int),
                "range_err_grid_m": float(r_grid - tg.range_m), "range_err_interp_m": float(r_int - tg.range_m),
                "vel_err_grid_mps": float(v_grid - tg.velocity_mps), "vel_err_interp_mps": float(v_int - tg.velocity_mps),
                "peak_snr_db": float(10 * np.log10(np.abs(RD[k, l]) ** 2 / noise_floor)),
                "expected_peak_snr_db_on_grid": float(tg.snr_db + 10 * np.log10(P.n_pulse_samples)
                                                       + 10 * np.log10(P.n_pulses if win == "rect" else
                                                                       np.sum(window("hann", P.n_pulses)) ** 2)),
            })
    arrays["rd_range_m"] = P.range_axis()
    arrays["rd_velocity_mps"] = doppler_axis(P.prf_hz, P.n_pulses) * P.wavelength / 2
    # slow-time cut through target 0 for rect vs hann
    l0 = int(round(2 * TARGETS[0].range_m / P.c * P.fs_hz))
    arrays["cut_range_index"] = np.array(l0)
    out["estimates"] = est
    out["true_targets"] = [t.__dict__ for t in TARGETS]
    return out, arrays


def _binning():
    """Noiseless sweeps of the fractional offset of range and Doppler from the grid."""
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    out, arrays = {}, {}
    frac = np.linspace(0.0, 1.0, 41)
    base_l = 300
    rg_err, ri_err, r_loss = [], [], []
    for f in frac:
        r = (base_l + f) * P.range_bin
        X = simulate_cpi(P, [PointTarget(r, 0.0, 0.0)], include_range_migration=False)
        y = matched_filter(X[0], h)
        lg = estimate_delay_samples(y, interpolate=False)
        li = estimate_delay_samples(y, interpolate=True)
        rg_err.append((lg - base_l - f))
        ri_err.append((li - base_l - f))
        r_loss.append(20 * np.log10(np.abs(y).max() / np.sqrt(h.size)))
    arrays["bin_frac"] = frac
    arrays["range_err_grid_bins"] = np.array(rg_err)
    arrays["range_err_interp_bins"] = np.array(ri_err)
    arrays["range_straddle_db"] = np.array(r_loss)
    dt = np.minimum(frac, 1 - frac) / P.fs_hz
    arrays["range_straddle_closed_db"] = 20 * np.log10(lfm_autocorrelation_magnitude(dt, P.bandwidth_hz, P.pulse_s))

    # Doppler: pure slow-time tone; isolates the FFT from range processing
    M = P.n_pulses
    m = np.arange(M)
    for win in ("rect", "hann"):
        g_err, i_err, loss, zp_err = [], [], [], []
        for f in frac:
            k_true = 10 + f
            s = np.exp(2j * np.pi * k_true * m / M)
            S = np.abs(np.fft.fft(s * window(win, M)))
            k = int(np.argmax(S))
            g_err.append(k - k_true)
            i_err.append(k + parabolic_peak(S, k) - k_true)
            loss.append(20 * np.log10(S.max() / np.abs(np.sum(window(win, M)))))
            S8 = np.abs(np.fft.fft(s * window(win, M), 8 * M))
            k8 = int(np.argmax(S8))
            zp_err.append((k8 + parabolic_peak(S8, k8)) / 8 - k_true)
        arrays[f"dop_{win}_err_grid_bins"] = np.array(g_err)
        arrays[f"dop_{win}_err_interp_bins"] = np.array(i_err)
        arrays[f"dop_{win}_err_zp8_interp_bins"] = np.array(zp_err)
        arrays[f"dop_{win}_loss_db"] = np.array(loss)
        arrays[f"dop_{win}_loss_closed_db"] = np.array([scalloping_loss_db(win, M, min(f, 1 - f)) for f in frac])
        out[f"doppler_{win}"] = {
            "max_abs_err_grid_bins": float(np.max(np.abs(g_err))),
            "max_abs_err_parabolic_bins": float(np.max(np.abs(i_err))),
            "max_abs_err_zeropad8_parabolic_bins": float(np.max(np.abs(zp_err))),
            "worst_scalloping_loss_db": float(np.min(loss)),
            "worst_scalloping_loss_closed_form_db": float(scalloping_loss_db(win, M, 0.5)),
            "max_abs_loss_dev_vs_closed_db": float(np.max(np.abs(np.array(loss) - arrays[f"dop_{win}_loss_closed_db"]))),
        }
    out["range"] = {
        "max_abs_err_grid_bins": float(np.max(np.abs(rg_err))),
        "max_abs_err_grid_m": float(np.max(np.abs(rg_err)) * P.range_bin),
        "max_abs_err_parabolic_bins": float(np.max(np.abs(ri_err))),
        "max_abs_err_parabolic_m": float(np.max(np.abs(ri_err)) * P.range_bin),
        "rms_err_grid_m": float(np.sqrt(np.mean(np.array(rg_err) ** 2)) * P.range_bin),
        "uniform_quantisation_rms_m": float(P.range_bin / np.sqrt(12)),
        "worst_straddle_loss_db": float(np.min(r_loss)),
        "worst_straddle_loss_closed_db": float(np.min(arrays["range_straddle_closed_db"])),
        "max_abs_straddle_dev_vs_closed_db": float(np.max(np.abs(np.array(r_loss) - arrays["range_straddle_closed_db"]))),
    }
    return out, arrays


def _estimation_mc(rng, snrs_db=(-30.0, -25.0, -20.0, -15.0, -10.0), n_trials=150):
    """RMS range/velocity error vs SNR for random sub-bin target positions (rect window)."""
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    fd_axis = doppler_axis(P.prf_hz, P.n_pulses)
    out, arrays = {"snr_per_sample_db": list(snrs_db)}, {}
    keys = ("r_grid", "r_interp", "v_grid", "v_interp")
    rms = {k: [] for k in keys}
    for snr in snrs_db:
        err = {k: [] for k in keys}
        for _ in range(n_trials):
            r = 5000.0 + rng.uniform(0, P.range_bin)
            v = 50.0 + rng.uniform(0, P.velocity_bin)
            X = simulate_cpi(P, [PointTarget(r, v, snr, rng.uniform(0, 2 * np.pi))], 1.0, rng)
            RD = np.abs(doppler_process(matched_filter(X, h, axis=1), "rect"))
            l0 = int(round(2 * r / P.c * P.fs_hz))
            k0 = int(np.argmin(np.abs(fd_axis - P.doppler_hz(v))))
            sub = RD[k0 - 3:k0 + 4, l0 - 3:l0 + 4]
            dk, dl = np.unravel_index(np.argmax(sub), sub.shape)
            k, l = k0 - 3 + dk, l0 - 3 + dl
            err["r_grid"].append(l * P.range_bin - r)
            err["r_interp"].append((l + parabolic_peak(RD[k, :], l)) * P.range_bin - r)
            err["v_grid"].append(fd_axis[k] * P.wavelength / 2 - v)
            err["v_interp"].append((fd_axis[k] + parabolic_peak(RD[:, l], k) * P.doppler_bin) * P.wavelength / 2 - v)
        for kk in keys:
            rms[kk].append(float(np.sqrt(np.mean(np.square(err[kk])))))
    for kk in keys:
        arrays[f"mc_rms_{kk}"] = np.array(rms[kk])
    arrays["mc_snr_db"] = np.array(snrs_db)
    arrays["mc_post_snr_db"] = np.array(snrs_db) + 10 * np.log10(P.n_pulse_samples * P.n_pulses)
    out["post_processing_snr_db_on_grid"] = arrays["mc_post_snr_db"].tolist()
    out["rms"] = rms
    out["quantisation_rms_range_m"] = P.range_bin / np.sqrt(12)
    out["quantisation_rms_velocity_mps"] = P.velocity_bin / np.sqrt(12)
    out["n_trials_per_snr"] = n_trials
    return out, arrays


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    m1, a1 = _mf_tests(rng)
    m2, a2 = _rd_map(rng)
    m3, a3 = _binning()
    m4, a4 = _estimation_mc(rng)
    metrics = {
        "derived_parameters": {
            "wavelength_m": P.wavelength, "range_bin_m": P.range_bin, "range_resolution_m": P.range_resolution,
            "unambiguous_range_m": P.unambiguous_range, "doppler_bin_hz": P.doppler_bin,
            "velocity_bin_mps": P.velocity_bin, "unambiguous_velocity_mps": P.unambiguous_velocity,
            "time_bandwidth": P.time_bandwidth, "pulse_samples": P.n_pulse_samples,
            "range_migration_over_cpi_m_at_250mps": 250 * P.n_pulses * P.pri,
        },
        "matched_filter": m1,
        "range_doppler_map": m2,
        "binning": m3,
        "estimation_monte_carlo": m4,
    }
    config = {"radar": P.__dict__, "targets": [t.__dict__ for t in TARGETS], "noise_power": 1.0, "seed": SEED}
    save_results(NAME, config, metrics, {**a1, **a2, **a3, **a4}, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, INK, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    m = rec["metrics"]
    paths = []

    # ---------------- Figure 7: matched filter ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(z["mf_ref_lag"], 20 * np.log10(np.maximum(z["mf_ref"], 1e-6)), color=C[0], lw=1.3,
            label=r"closed form $|\chi(\tau,0)|/T$ at true delay")
    ax.plot(z["mf_lag"], 20 * np.log10(np.maximum(z["mf_mag"], 1e-6)), "o", ms=4, mfc="none", color=C[1],
            label="sampled matched-filter output")
    tau0 = 2 * rec["config"]["targets"][0]["range_m"] / 299792458.0 * 10e6
    ax.axvline(tau0, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(-45, 3)
    ax.set_xlabel("fast-time lag (samples, 1 sample = %.2f m)" % m["derived_parameters"]["range_bin_m"])
    ax.set_ylabel(r"$|y|/\sqrt{L}$ (dB)")
    ax.set_title("(a) LFM compression, noiseless, fractional delay", loc="left")
    ax.legend(loc="upper left", fontsize=7.2)

    ax = axes[1]
    tb = z["res_tau_x_B"]
    for i, (key, lab) in enumerate([("0", "in phase"), ("pi_2", "quadrature"), ("pi", "anti-phase")]):
        ax.plot(tb, 20 * np.log10(np.maximum(z[f"res_1.0_phase_{key}"], 1e-6)), color=C[i],
                ls=["-", "--", ":"][i], label=lab)
    for xv in (0.0, 1.0):
        ax.axvline(xv, color=MUTED, lw=0.8, ls="-.")
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-30, 9)
    ax.set_xlabel(r"delay $\times B$ (target 1 at 0, target 2 at 1)")
    ax.set_ylabel(r"$|y|/\sqrt{L}$ (dB)")
    ax.set_title(r"(b) Two equal targets, $\Delta R = c/(2B)$", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[2]
    ax.plot(z["coupling_fd_hz"] / 1e3, z["coupling_pred_samples"], color=C[0], label=r"$-f_D T/B$ (theory)")
    ax.plot(z["coupling_fd_hz"] / 1e3, z["coupling_meas_samples"], "o", ms=4, mfc="none", color=C[1],
            label="measured MF peak shift")
    ax.set_xlabel(r"Doppler shift $f_D$ (kHz)")
    ax.set_ylabel("peak delay shift (samples)")
    ax.set_title("(c) LFM range–Doppler coupling", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.axvspan(-5, 5, color=MUTED, alpha=0.2, lw=0)
    ax.text(8, -8, "shaded: unambiguous\nDoppler (±PRF/2)", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    paths.append(save(fig, "fig07_matched_filter.png"))

    # ---------------- Figure 8: range-Doppler map ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    rd = z["rd_hann_db"].astype(float)
    noise_db = 10 * np.log10(np.median(10 ** (rd / 10)) / np.log(2))
    rng_m, vel = z["rd_range_m"], z["rd_velocity_mps"]
    dv = vel[1] - vel[0]
    ax = axes[0]
    pool = 5  # display only: max over 5 adjacent range cells so single-cell peaks survive rendering
    nr = (rd.shape[1] // pool) * pool
    pooled = rd[:, :nr].reshape(rd.shape[0], -1, pool).max(axis=2)
    im = ax.imshow(pooled - noise_db, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=25,
                   interpolation="nearest",
                   extent=[rng_m[0] / 1e3, rng_m[nr - 1] / 1e3, vel[0] - dv / 2, vel[-1] + dv / 2])
    for t in rec["config"]["targets"]:
        ax.plot(t["range_m"] / 1e3, t["velocity_mps"], marker="o", ms=13, mfc="none", mec="white", mew=1.2)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("power re: noise floor (dB)")
    ax.set_xlabel("range (km)")
    ax.set_ylabel("radial velocity (m/s, + = closing)")
    ax.set_title("(a) Full map, Hann window (display: max over 5 range cells)", loc="left")
    ax.grid(False)

    ax = axes[1]
    l0 = int(z["cut_range_index"])
    sl = slice(l0 - 25, l0 + 26)
    im2 = ax.imshow(rd[:, sl] - noise_db, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=25,
                    interpolation="nearest",
                    extent=[rng_m[sl][0] / 1e3 - 0.0075, rng_m[sl][-1] / 1e3 + 0.0075, vel[0] - dv / 2, vel[-1] + dv / 2])
    t0 = rec["config"]["targets"][0]
    ax.plot(t0["range_m"] / 1e3, t0["velocity_mps"], marker="+", ms=12, color="white", mew=1.2)
    ax.set_xlabel("range (km)")
    ax.set_ylabel("radial velocity (m/s)")
    ax.set_title("(b) Zoom on target 1, every cell (+: injected)", loc="left")
    ax.grid(False)

    ax = axes[2]
    for win, col, ls in (("rect", C[0], "-"), ("hann", C[1], "--")):
        cut = z[f"rd_{win}_db"].astype(float)[:, l0]
        nd = 10 * np.log10(np.median(10 ** (z[f"rd_{win}_db"].astype(float) / 10)) / np.log(2))
        ax.plot(vel, cut - nd, color=col, ls=ls, drawstyle="steps-mid", label=f"{win} window")
    ax.axvline(t0["velocity_mps"], color=INK, lw=0.8, ls=":")
    ax.set_xlabel("radial velocity (m/s)")
    ax.set_ylabel("power re: noise floor (dB)")
    ax.set_title(f"(c) Doppler cut at R = {rng_m[l0] / 1e3:.3f} km", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    paths.append(save(fig, "fig08_range_doppler_map.png"))

    # ---------------- Figure 9: binning ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    axes = axes.ravel()
    f = z["bin_frac"]
    ax = axes[0]
    ax.plot(f, z["range_err_grid_bins"], color=C[0], label="grid argmax")
    ax.plot(f, z["range_err_interp_bins"], color=C[1], ls="--", marker="o", ms=3, mfc="none",
            label=r"3-point parabolic on $|y|$")
    ax.set_xlabel("true delay offset from sample grid (samples)")
    ax.set_ylabel("range error (samples)")
    ax.set_title(f"(a) Range estimate error (1 sample = {m['derived_parameters']['range_bin_m']:.2f} m)", loc="left")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    for win, col in (("rect", C[0]), ("hann", C[2])):
        ax.plot(f, z[f"dop_{win}_err_grid_bins"], color=col, label=f"{win}: grid argmax")
        ax.plot(f, z[f"dop_{win}_err_interp_bins"], color=col, ls="--", marker="o", ms=3, mfc="none",
                label=f"{win}: parabolic")
        ax.plot(f, z[f"dop_{win}_err_zp8_interp_bins"], color=col, ls=":", marker="^", ms=3, mfc="none",
                label=f"{win}: 8× zero-pad + parabolic")
    ax.set_xlabel("true Doppler offset from bin center (bins)")
    ax.set_ylabel("Doppler error (bins)")
    ax.set_title(f"(b) Doppler estimate error (1 bin = {m['derived_parameters']['velocity_bin_mps']:.2f} m/s)",
                 loc="left")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[2]
    ax.plot(f, z["range_straddle_closed_db"], color=C[1], lw=1.2, label="range straddle, closed form")
    ax.plot(f, z["range_straddle_db"], "o", ms=3.5, mfc="none", color=C[1], label="range straddle, measured")
    for win, col in (("rect", C[0]), ("hann", C[2])):
        ax.plot(f, z[f"dop_{win}_loss_closed_db"], color=col, lw=1.2, label=f"Doppler scalloping ({win}), exact sum")
        ax.plot(f, z[f"dop_{win}_loss_db"], "s", ms=3.5, mfc="none", color=col, label=f"Doppler scalloping ({win}), FFT")
    ax.set_xlabel("offset from grid (samples or bins)")
    ax.set_ylabel("peak loss (dB)")
    ax.set_title("(c) Straddle and scalloping loss", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(-4.5, 0.3)

    ax = axes[3]
    e = m["estimation_monte_carlo"]
    x = z["mc_post_snr_db"]
    ax.semilogy(x, z["mc_rms_r_grid"] / m["derived_parameters"]["range_bin_m"], "o-", color=C[0],
                label="range, grid argmax")
    ax.semilogy(x, z["mc_rms_r_interp"] / m["derived_parameters"]["range_bin_m"], "o--", mfc="none", color=C[0],
                label="range, parabolic")
    ax.semilogy(x, z["mc_rms_v_grid"] / m["derived_parameters"]["velocity_bin_mps"], "s-", color=C[1],
                label="velocity, grid argmax")
    ax.semilogy(x, z["mc_rms_v_interp"] / m["derived_parameters"]["velocity_bin_mps"], "s--", mfc="none",
                color=C[1], label="velocity, parabolic")
    ax.axhline(1 / np.sqrt(12), color=MUTED, ls=":", lw=1)
    ax.text(x[0], 1 / np.sqrt(12) * 1.08, r"uniform quantization 1/$\sqrt{12}$", fontsize=7.5, color=MUTED)
    ax.set_xlabel("post-processing SNR for a grid-centered target (dB)")
    ax.set_ylabel("RMS error (range samples or Doppler bins)")
    ax.set_title(f"(d) Monte Carlo RMS error, {e['n_trials_per_snr']} trials/point, rect window", loc="left")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    paths.append(save(fig, "fig09_binning_effects.png"))
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        print(json.dumps(run(), indent=1, default=str))
    print(plot())

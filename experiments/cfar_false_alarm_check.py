"""Part E - CFAR threshold example and Monte Carlo false-alarm verification.

1. Threshold example (figure 10): a range cut of a simulated range-Doppler map
   with a weak target next to a strong one; CA- and OS-CFAR thresholds.
2. i.i.d. exponential noise (figure 11a): measured vs requested P_FA for
   CA-CFAR (N_t = 16, 32) and OS-CFAR (N_t = 16, k = 12).
3. Noise that has passed through the actual processing chain (figure 11b):
   matched filtering at f_s = 2B correlates neighboring range cells, and a
   Hann slow-time window correlates neighboring Doppler cells. The CA-CFAR
   threshold formula assumes independent training cells; the effect of the
   correlation on P_FA is measured and compared with a moment-matching
   prediction based on an effective number of independent training cells.

Statistical method: decisions in one sequence/map are correlated (shared
training cells), so confidence intervals use the spread of the per-sequence
(or per-map) false-alarm fraction over independent sequences (maps).

Usage: ``python experiments/cfar_false_alarm_check.py [--plot-only]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from array_signal_lab.cfar import ca_alpha, ca_cfar_1d, ca_cfar_2d, ca_pfa, os_cfar_1d
from array_signal_lab.covariance import complex_gaussian
from array_signal_lab.doppler import doppler_process, window
from array_signal_lab.io import load_results, save_results
from array_signal_lab.matched_filter import matched_filter
from array_signal_lab.radar_signal import PointTarget, RadarParams, simulate_cpi
from array_signal_lab.waveforms import lfm_pulse

NAME = "cfar_false_alarm_check"
SEED = 20261008
PFA_LIST = [1e-2, 1e-3, 1e-4, 1e-5]
SEQ_LEN = 10_000
N_SEQ = 2_000  # 2e7 cells per configuration


def _mean_ci(fracs):
    fracs = np.asarray(fracs, dtype=float)
    m = fracs.mean()
    se = fracs.std(ddof=1) / np.sqrt(fracs.size)
    return float(m), float(se)


def _iid_study(rng):
    rows = []
    configs = [("CA", 8, 2, None), ("CA", 16, 2, None), ("OS", 8, 2, 12)]
    for kind, ntr, ngd, k in configs:
        for pfa in PFA_LIST:
            if kind == "OS" and pfa < 1e-4:
                continue
            n_seq = N_SEQ if kind == "CA" else 400
            fr = []
            for c0 in range(0, n_seq, 100):
                z = rng.exponential(size=(min(100, n_seq - c0), SEQ_LEN))
                if kind == "CA":
                    det, thr = ca_cfar_1d(z, ntr, ngd, pfa, "valid")
                else:
                    det, thr = os_cfar_1d(z, ntr, ngd, k, pfa)
                nvalid = np.isfinite(thr).sum(axis=1)
                fr.append(det.sum(axis=1) / nvalid)
            fr = np.concatenate(fr)
            m, se = _mean_ci(fr)
            n_cells = int(n_seq * (SEQ_LEN - 2 * (ntr + ngd)))
            rows.append({
                "detector": f"{kind}-CFAR N_t={2 * ntr}" + (f", k={k}" if k else ""),
                "pfa_requested": pfa, "pfa_measured": m, "stderr": se,
                "ci95": [m - 1.96 * se, m + 1.96 * se],
                "z_score": (m - pfa) / se if se > 0 else None,
                "ratio_measured_to_requested": m / pfa,
                "n_cells": n_cells, "n_false_alarms": int(round(m * n_cells)),
                "binomial_stderr_if_independent": float(np.sqrt(pfa * (1 - pfa) / n_cells)),
            })
    # Replicate any configuration whose |z| exceeds 2.5 with an independent random stream,
    # to distinguish a statistical fluctuation from a systematic error.
    rng2 = np.random.default_rng(SEED + 1)
    for r in rows:
        if r["z_score"] is not None and abs(r["z_score"]) > 2.5 and r["detector"].startswith("CA"):
            ntr = int(r["detector"].split("N_t=")[1]) // 2
            fr = []
            for _ in range(0, N_SEQ, 100):
                z = rng2.exponential(size=(100, SEQ_LEN))
                det, thr = ca_cfar_1d(z, ntr, 2, r["pfa_requested"], "valid")
                fr.append(det.sum(axis=1) / np.isfinite(thr).sum(axis=1))
            m, se = _mean_ci(np.concatenate(fr))
            r["replication"] = {"seed": SEED + 1, "pfa_measured": m, "stderr": se,
                                "z_score": (m - r["pfa_requested"]) / se}
    return rows


def _field_corr_range(replica, max_lag):
    """Complex correlation coefficient of white noise after the matched filter vs lag."""
    L = replica.size
    h = replica / np.linalg.norm(replica)
    return np.array([np.vdot(h[l:], h[:L - l]) if l < L else 0.0 for l in range(max_lag + 1)])


def _field_corr_doppler(win_name, m, max_lag):
    g = window(win_name, m)
    n = np.arange(m)
    return np.array([np.sum(g**2 * np.exp(2j * np.pi * d * n / m)) for d in range(max_lag + 1)])


def _neff(rho_k, rho_l, n_train, n_guard):
    """Effective number of independent training cells, N_t^2 / sum_ij |rho_ij|^2, for the 2-D window.

    For complex Gaussian fields the power correlation is |rho|^2; rho is separable
    (Doppler x range).
    """
    tk, tl = n_train
    gk, gl = n_guard
    rk, rl = tk + gk, tl + gl
    cells = [(a, b) for a in range(-rk, rk + 1) for b in range(-rl, rl + 1) if abs(a) > gk or abs(b) > gl]
    cells = np.array(cells)
    da = np.abs(cells[:, 0][:, None] - cells[:, 0][None, :])
    db = np.abs(cells[:, 1][:, None] - cells[:, 1][None, :])
    rk_full = np.pad(rho_k, (0, 2 * rk + 1))
    rl_full = np.pad(rho_l, (0, 2 * rl + 1))
    r2 = np.abs(rk_full[da]) ** 2 * np.abs(rl_full[db]) ** 2
    nt = len(cells)
    return float(nt**2 / r2.sum()), nt


def _pipeline_study(rng, n_maps=60, pfas=(1e-3, 1e-4)):
    train, guard = (2, 8), (1, 2)
    cases = [
        ("iid exponential (control)", None, None),
        ("MF f_s = 2B, rect Doppler", 10e6, "rect"),
        ("MF f_s = 2B, Hann Doppler", 10e6, "hann"),
        ("MF f_s = B, rect Doppler", 5e6, "rect"),
        ("MF f_s = B, Hann Doppler", 5e6, "hann"),
    ]
    rows = []
    edge_rows = []
    for name, fs, win in cases:
        P = RadarParams(fs_hz=fs or 10e6)
        h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
        L = h.size
        full_end = P.n_fast - L  # lags >= full_end see a partial pulse
        fr = {p: [] for p in pfas}
        fr_edge = {p: [] for p in pfas}
        for _ in range(n_maps):
            if fs is None:
                z = rng.exponential(size=(P.n_pulses, P.n_fast))
            else:
                X = complex_gaussian(rng, (P.n_pulses, P.n_fast), 1.0)
                z = np.abs(doppler_process(matched_filter(X, h, axis=1), win)) ** 2
            for p in pfas:
                det, thr, nt = ca_cfar_2d(z, train, guard, p, wrap_axis0=True, edge="truncate")
                core = slice(train[1] + guard[1], full_end - (train[1] + guard[1]))
                fr[p].append(det[:, core].mean())
                if fs is not None:
                    fr_edge[p].append(det[:, full_end:].mean())
        # moment-matching prediction
        if fs is None:
            rho_l = np.array([1.0])
            rho_k = np.array([1.0])
        else:
            rho_l = _field_corr_range(h, 2 * (train[1] + guard[1]))
            rho_k = _field_corr_doppler(win, P.n_pulses, 2 * (train[0] + guard[0]))
        neff, nt = _neff(rho_k, rho_l, train, guard)
        for p in pfas:
            m, se = _mean_ci(fr[p])
            alpha = ca_alpha(nt, p)
            pred = ca_pfa(alpha, neff)  # (1 + alpha/N_eff)^(-N_eff)
            rows.append({
                "case": name, "pfa_requested": p, "pfa_measured": m, "stderr": se,
                "z_score": (m - p) / se, "ratio_measured_to_requested": m / p,
                "n_train_cells": nt, "n_eff": neff, "pfa_predicted_neff": float(pred),
                "rho_range_lag1_abs": float(np.abs(rho_l[1])) if rho_l.size > 1 else 0.0,
                "rho_doppler_lag1_abs": float(np.abs(rho_k[1])) if rho_k.size > 1 else 0.0,
                "n_maps": n_maps,
            })
            if fs is not None:
                me, see = _mean_ci(fr_edge[p])
                edge_rows.append({"case": name, "pfa_requested": p, "pfa_measured_partial_overlap_lags": me,
                                  "stderr": see, "lags": [int(full_end), int(P.n_fast - 1)]})
    return rows, edge_rows, {"train_doppler_range_per_side": train, "guard_doppler_range_per_side": guard}


def _threshold_example(rng):
    P = RadarParams()
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    v = 101.3
    targets = [PointTarget(4321.7, v, -15.0, 0.3), PointTarget(4321.7 + 9 * P.range_bin, v, -10.0, 1.0),
               PointTarget(9007.4, v, -20.0, 2.0)]
    X = simulate_cpi(P, targets, 1.0, rng)
    RD = doppler_process(matched_filter(X, h, axis=1), "hann")
    fd = np.fft.fftshift(np.fft.fftfreq(P.n_pulses, P.pri))
    k = int(np.argmin(np.abs(fd - P.doppler_hz(v))))
    zc = np.abs(RD[k]) ** 2
    pfa = 1e-4
    d_ca, t_ca = ca_cfar_1d(zc, 8, 2, pfa, "valid")
    d_tr, t_tr = ca_cfar_1d(zc, 8, 2, pfa, "truncate")
    d_os, t_os = os_cfar_1d(zc, 8, 2, 12, pfa)
    idx = [int(round(2 * t.range_m / P.c * P.fs_hz)) for t in targets]
    out = {
        "pfa": pfa, "doppler_bin_index": k, "target_lags": idx,
        "targets": [t.__dict__ for t in targets],
        "ca_detected_targets": [bool(d_ca[i - 1:i + 2].any()) for i in idx],
        "os_detected_targets": [bool(d_os[i - 1:i + 2].any()) for i in idx],
        "ca_n_detections": int(d_ca.sum()), "os_n_detections": int(d_os.sum()),
        "ca_threshold_over_noise_db_at_weak_target": float(10 * np.log10(t_ca[idx[0]] / np.median(zc) * np.log(2))),
        "os_threshold_over_noise_db_at_weak_target": float(10 * np.log10(t_os[idx[0]] / np.median(zc) * np.log(2))),
        "weak_target_cell_power_over_noise_db": float(10 * np.log10(zc[idx[0] - 1:idx[0] + 2].max() / np.median(zc) * np.log(2))),
    }
    arrays = {"ex_z": zc, "ex_thr_ca": t_ca, "ex_thr_trunc": t_tr, "ex_thr_os": t_os,
              "ex_det_ca": d_ca, "ex_det_os": d_os, "ex_range_m": P.range_axis(), "ex_target_lags": np.array(idx)}
    return out, arrays


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    ex, arrays = _threshold_example(rng)
    iid = _iid_study(rng)
    pipe, edge, geom = _pipeline_study(rng)
    for key, rows in (("iid", iid), ("pipe", pipe)):
        arrays[f"{key}_req"] = np.array([r["pfa_requested"] for r in rows])
        arrays[f"{key}_meas"] = np.array([r["pfa_measured"] for r in rows])
        arrays[f"{key}_se"] = np.array([r["stderr"] for r in rows])
    arrays["iid_label"] = np.array([r["detector"] for r in iid])
    arrays["pipe_label"] = np.array([r["case"] for r in pipe])
    arrays["pipe_pred"] = np.array([r["pfa_predicted_neff"] for r in pipe])
    metrics = {"threshold_example": ex, "iid": iid, "pipeline": pipe, "pipeline_partial_overlap_edge": edge,
               "pipeline_window": geom,
               "max_abs_z_iid": float(max(abs(r["z_score"]) for r in iid)),
               "iid_chi2_sum_z2": float(sum(r["z_score"] ** 2 for r in iid)),
               "iid_chi2_dof": len(iid),
               "iid_chi2_p_value": float(__import__("scipy.stats", fromlist=["chi2"]).chi2.sf(
                   sum(r["z_score"] ** 2 for r in iid), len(iid)))}
    config = {"pfa_list": PFA_LIST, "iid_seq_len": SEQ_LEN, "iid_n_seq_CA": N_SEQ, "iid_n_seq_OS": 400,
              "iid_window_per_side": {"CA16": [8, 2], "CA32": [16, 2], "OS16": [8, 2, 12]},
              "pipeline_maps_per_case": 60, "seed": SEED}
    save_results(NAME, config, metrics, arrays, elapsed_s=time.perf_counter() - t0)
    return metrics


def plot():
    import matplotlib.pyplot as plt
    from plotstyle import C, INK, MUTED, save, setup

    setup()
    rec, z = load_results(NAME)
    m = rec["metrics"]
    paths = []

    # ---------------- Figure 10: threshold example ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    r = z["ex_range_m"] / 1e3
    noise = np.median(z["ex_z"]) / np.log(2)
    zdb = 10 * np.log10(z["ex_z"] / noise)
    for ax, sl, title in ((axes[0], slice(None), "(a) Range cut at the target Doppler bin, P$_{FA}$ = 10$^{-4}$"),
                          (axes[1], slice(270, 330), "(b) Zoom: weak target masked by strong neighbor")):
        ax.plot(r[sl], zdb[sl], color=MUTED, lw=0.8, label="cell power |y|²")
        ax.plot(r[sl], 10 * np.log10(z["ex_thr_ca"][sl] / noise), color=C[0], lw=1.5, label="CA-CFAR threshold (valid)")
        ax.plot(r[sl], 10 * np.log10(z["ex_thr_trunc"][sl] / noise), color=C[0], lw=1.0, ls=":",
                label="CA-CFAR, truncated window at edges")
        ax.plot(r[sl], 10 * np.log10(z["ex_thr_os"][sl] / noise), color=C[1], lw=1.4, ls="--",
                label="OS-CFAR threshold (k = 12 of 16)")
        dca = np.where(z["ex_det_ca"][sl])[0]
        dos = np.where(z["ex_det_os"][sl])[0]
        ax.plot(r[sl][dca], zdb[sl][dca], "o", ms=7, mfc="none", color=C[0], label="CA detections")
        ax.plot(r[sl][dos], zdb[sl][dos], "x", ms=6, color=C[1], label="OS detections")
        ax.set_xlabel("range (km)")
        ax.set_ylabel("power re: noise (dB)")
        ax.set_title(title, loc="left")
        ax.set_ylim(-25, 35)
    for li in z["ex_target_lags"][:2]:
        axes[1].axvline(r[li], color=INK, lw=0.7, ls="-.")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    paths.append(save(fig, "fig10_cfar_threshold_example.png"))

    # ---------------- Figure 11: Pfa verification ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    labels = list(dict.fromkeys(z["iid_label"].tolist()))
    mk = ["o", "s", "^"]
    for i, lab in enumerate(labels):
        sel = z["iid_label"] == lab
        req = z["iid_req"][sel]
        ax.errorbar(req * (1 + 0.12 * (i - 1)), z["iid_meas"][sel] / req, yerr=1.96 * z["iid_se"][sel] / req,
                    fmt=mk[i], color=C[i], mfc="none", ms=6, capsize=3, label=lab)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0.75, 1.25)
    ax.set_xlabel("requested P$_{FA}$")
    ax.set_ylabel("measured / requested P$_{FA}$ (95 % CI)")
    ax.set_title("(a) i.i.d. exponential noise, 1-D window, 2·10$^7$ cells per CA point", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    pl = z["pipe_label"]
    cases = list(dict.fromkeys(pl.tolist()))
    for j, p in enumerate((1e-3, 1e-4)):
        sel = z["pipe_req"] == p
        y = np.arange(len(cases)) + (0.18 if j == 0 else -0.18)
        ratio = z["pipe_meas"][sel] / p
        err = 1.96 * z["pipe_se"][sel] / p
        ax.errorbar(ratio, y, xerr=err, fmt="o", color=C[j], ms=6, capsize=3,
                    label=f"measured, requested {p:.0e}")
        ax.plot(z["pipe_pred"][sel] / p, y, "D", mfc="none", color=C[j], ms=8, mew=1.3,
                label=f"N$_{{eff}}$ prediction, {p:.0e}")
    ax.axvline(1.0, color=MUTED, lw=1, ls=":")
    ax.set_xlim(0.5, 2.2)
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels(cases, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("measured / requested P$_{FA}$ (95 % CI)")
    ax.set_title("(b) 2-D CA-CFAR on processed noise (N$_t$ = %d)" % rec["metrics"]["pipeline"][0]["n_train_cells"],
                 loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    paths.append(save(fig, "fig11_cfar_pfa_verification.png"))
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        import json
        print(json.dumps(run(), indent=1, default=str))
    print(plot())

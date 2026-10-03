"""Shared scenario definition and Monte Carlo helper for the beamforming studies (Parts B and C)."""

from __future__ import annotations

import numpy as np

from array_signal_lab import (
    SingularCovarianceError,
    ULA,
    mvdr_weights,
    sample_covariance,
    simulate_snapshots,
    steering_vector,
    theoretical_covariance,
)
from array_signal_lab.metrics import optimal_sinr, output_sinr

# Generic baseline scenario (same as jammer_suppression.py)
N = 16
THETA_S_DEG = 0.0
JAMMERS_DEG = (35.0, -42.0)
JNR_DB = (40.0, 30.0)
SNR_DB = 0.0  # per element


def baseline():
    ula = ULA(N)
    th_s = np.deg2rad(THETA_S_DEG)
    th_j = np.deg2rad(JAMMERS_DEG)
    jnr = 10 ** (np.array(JNR_DB) / 10)
    snr = 10 ** (SNR_DB / 10)
    a_s = steering_vector(ula, th_s)
    R_in = theoretical_covariance(ula, th_j, jnr, 1.0)
    return dict(ula=ula, th_s=th_s, th_j=th_j, jnr=jnr, snr=snr, a_s=a_s, R_in=R_in,
                sinr_opt=optimal_sinr(a_s, snr, R_in))


def smi_trials(sc, k, n_trials, rng, signal_in_training=False, loading=0.0, a_look=None,
               th_s_true=None, extra=None):
    """Monte Carlo over independent training sets.

    Returns an array of normalized SINR ``rho = SINR(w_hat)/SINR_opt`` (NaN where the
    unloaded covariance was singular) and the fraction of singular trials.

    ``a_look``: steering vector used in the MVDR constraint (default: true target).
    ``th_s_true``: true target angle (default: scenario target).
    ``extra``: optional callable ``f(w) -> float`` evaluated per trial.
    """
    ula = sc["ula"]
    th_s = sc["th_s"] if th_s_true is None else th_s_true
    a_s = steering_vector(ula, th_s)
    a_look = a_s if a_look is None else a_look
    sinr_opt = optimal_sinr(a_s, sc["snr"], sc["R_in"])
    if signal_in_training:
        angles, powers = np.r_[th_s, sc["th_j"]], np.r_[sc["snr"], sc["jnr"]]
    else:
        angles, powers = sc["th_j"], sc["jnr"]
    rho = np.full(n_trials, np.nan)
    ex = np.full(n_trials, np.nan)
    n_sing = 0
    for t in range(n_trials):
        X = simulate_snapshots(ula, angles, powers, 1.0, k, rng)
        try:
            w = mvdr_weights(sample_covariance(X), a_look, loading=loading)
        except SingularCovarianceError:
            n_sing += 1
            continue
        rho[t] = output_sinr(w, a_s, sc["snr"], sc["R_in"]) / sinr_opt
        if extra is not None:
            ex[t] = extra(w)
    return rho, n_sing / n_trials, ex


def mean_ci_db(rho):
    """Mean of linear rho (NaNs dropped) and a 95 % normal-approximation CI, all in dB."""
    r = rho[np.isfinite(rho)]
    if r.size < 2:
        return np.nan, np.nan, np.nan
    m = r.mean()
    se = r.std(ddof=1) / np.sqrt(r.size)
    return 10 * np.log10(m), 10 * np.log10(max(m - 1.96 * se, 1e-300)), 10 * np.log10(m + 1.96 * se)

"""Beamformer performance metrics and closed-form reference values.

All SINR values are *linear* unless a function name ends in ``_db``.
Output SINR of a weight vector is always evaluated with the **true**
interference-plus-noise covariance ``R_in``, even when the weights were
computed from an estimate; this is the standard definition used in the
sample-matrix-inversion literature [Reed, Mallett & Brennan 1974].
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def db(x):
    """``10 log10(x)`` for power quantities."""
    return 10.0 * np.log10(x)


def from_db(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def output_sinr(w: np.ndarray, a_s: np.ndarray, signal_power: float, R_in: np.ndarray) -> float:
    """``p_s |w^H a_s|^2 / (w^H R_in w)``."""
    num = signal_power * np.abs(np.vdot(w, a_s)) ** 2
    den = np.real(np.vdot(w, R_in @ w))
    return float(num / den)


def optimal_sinr(a_s: np.ndarray, signal_power: float, R_in: np.ndarray) -> float:
    """Upper bound ``p_s a_s^H R_in^{-1} a_s`` (achieved by ``w ~ R_in^{-1} a_s``)."""
    return float(signal_power * np.real(np.vdot(a_s, np.linalg.solve(R_in, a_s))))


def optimal_sinr_single_jammer(n: int, snr: float, jnr: float, beta: complex) -> float:
    """Closed-form optimum SINR for one jammer in white noise.

    With ``R_in = sigma^2 (I + jnr a_j a_j^H)`` and ``beta = a_s^H a_j``,
    the matrix inversion lemma gives

        SINR_opt = snr * (N - jnr |beta|^2 / (1 + N jnr)).

    ``snr`` and ``jnr`` are per-element ratios to ``sigma^2``.
    """
    return float(snr * (n - jnr * np.abs(beta) ** 2 / (1.0 + n * jnr)))


def conventional_sinr_single_jammer(n: int, snr: float, jnr: float, beta: complex) -> float:
    """Closed-form SINR of ``w = a_s / N`` for one jammer in white noise.

        SINR_conv = snr * N^2 / (N + jnr |beta|^2).
    """
    return float(snr * n**2 / (n + jnr * np.abs(beta) ** 2))


def white_noise_gain(w: np.ndarray, a: np.ndarray) -> float:
    """``|w^H a|^2 / ||w||^2``; equals ``N`` for the conventional beamformer."""
    return float(np.abs(np.vdot(w, a)) ** 2 / np.real(np.vdot(w, w)))


def relative_response_db(w: np.ndarray, a_ref: np.ndarray, a_other: np.ndarray) -> float:
    """``20 log10 |w^H a_other| / |w^H a_ref|`` (e.g. jammer response relative to target)."""
    return float(20.0 * np.log10(np.abs(np.vdot(w, a_other)) / np.abs(np.vdot(w, a_ref))))


def rmb_mean_normalized_sinr(n_snapshots: int, n_elements: int) -> float:
    """Reed-Mallett-Brennan mean of ``rho = SINR(w_hat) / SINR_opt``.

    For SMI weights from ``K >= N`` signal-free, i.i.d. complex Gaussian
    training snapshots, ``rho ~ Beta(K - N + 2, N - 1)`` with mean
    ``(K - N + 2) / (K + 1)``.
    """
    k, n = n_snapshots, n_elements
    if k < n:
        return float("nan")
    return (k - n + 2.0) / (k + 1.0)


def rmb_distribution(n_snapshots: int, n_elements: int):
    """``scipy.stats.beta(K - N + 2, N - 1)`` frozen distribution of ``rho``."""
    return stats.beta(n_snapshots - n_elements + 2, n_elements - 1)


def smi_signal_present_mean_normalized_sinr(n_snapshots: int, n_elements: int, sinr_opt: float) -> float:
    """Mean of ``SINR(w_hat) / SINR_opt`` when the target is present in the training data.

    The RMB result applies to the *full* covariance ``R = R_in + p_s a a^H`` from
    which the training data are drawn: ``rho = |w^H a|^2 / (w^H R w a^H R^{-1} a)``
    is ``Beta(K - N + 2, N - 1)``. Using ``p_s a^H R^{-1} a = S / (1 + S)``
    (``S = SINR_opt``, matrix inversion lemma) and
    ``SINR_out = x / (1 - x)`` with ``x = p_s |w^H a|^2 / (w^H R w)``, one obtains

        SINR_out / S = rho / (1 + S (1 - rho)),

    whose mean is evaluated here by quadrature over the Beta density.
    """
    from scipy import integrate

    if n_snapshots < n_elements:
        return float("nan")
    dist = rmb_distribution(n_snapshots, n_elements)
    s = float(sinr_opt)
    val, _ = integrate.quad(lambda r: r / (1.0 + s * (1.0 - r)) * dist.pdf(r), 0.0, 1.0, limit=200)
    return float(val)


def generalized_cos2(a0: np.ndarray, a_s: np.ndarray, R_in: np.ndarray) -> float:
    """``cos^2(a0, a_s; R_in^{-1}) = |a0^H R^{-1} a_s|^2 / ((a0^H R^{-1} a0)(a_s^H R^{-1} a_s))``."""
    u0 = np.linalg.solve(R_in, a0)
    us = np.linalg.solve(R_in, a_s)
    return float(np.abs(np.vdot(a0, us)) ** 2 / (np.real(np.vdot(a0, u0)) * np.real(np.vdot(a_s, us))))


def mismatched_mvdr_sinr(a0, a_s, signal_power, R_in) -> float:
    """SINR of MVDR built from the *interference-plus-noise* covariance, steered to ``a0``
    while the target arrives along ``a_s``: ``SINR_opt * cos^2``."""
    return optimal_sinr(a_s, signal_power, R_in) * generalized_cos2(a0, a_s, R_in)


def mismatched_mpdr_sinr(a0, a_s, signal_power, R_in) -> float:
    """SINR of the minimum-power (MPDR) beamformer ``R^{-1} a0`` with the target *in* ``R``.

    With ``S = SINR_opt`` and ``c2 = cos^2(a0, a_s; R_in^{-1})`` [Cox 1973]:

        SINR = S c2 / (1 + (2 S + S^2)(1 - c2)).
    """
    s = optimal_sinr(a_s, signal_power, R_in)
    c2 = generalized_cos2(a0, a_s, R_in)
    return float(s * c2 / (1.0 + (2.0 * s + s**2) * (1.0 - c2)))

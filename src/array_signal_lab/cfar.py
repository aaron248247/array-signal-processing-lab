"""Constant-false-alarm-rate (CFAR) detection on square-law (power) data.

Noise model
-----------
Complex Gaussian noise ``CN(0, sigma^2)`` after a square-law detector gives
``z = |x|^2 ~ Exponential(mean sigma^2)``. All threshold formulas below assume
that the training cells and the cell under test (CUT) are independent and
identically distributed under H0.

Cell-averaging (CA) CFAR [Finn & Johnson 1968]
----------------------------------------------
With ``N_t`` training cells, ``Z = sum z_i ~ Gamma(N_t, sigma^2)`` and the test
``z_CUT > (alpha / N_t) Z``. Averaging ``exp(-T/sigma^2)`` over ``Z``:

    P_FA = (1 + alpha / N_t)^(-N_t)   =>   alpha = N_t (P_FA^(-1/N_t) - 1),

independent of ``sigma^2`` (the CFAR property).

Ordered-statistic (OS) CFAR [Rohling 1983]
------------------------------------------
Threshold ``alpha * z_(k)`` with ``z_(k)`` the k-th smallest training cell:

    P_FA = prod_{i=0}^{k-1} (N_t - i) / (N_t - i + alpha).

Window geometry (1-D, per side): ``n_guard`` guard cells adjacent to the CUT,
then ``n_train`` training cells. Edge modes:

* ``"valid"``    - no decision where the full window does not fit (threshold NaN).
* ``"truncate"`` - use whatever training cells exist; ``alpha`` is recomputed for
  the actual count, so P_FA stays nominal for i.i.d. noise.
* ``"wrap"``     - circular indexing (appropriate for the Doppler axis, which is
  periodic in PRF).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, stats


# --------------------------------------------------------------------------- thresholds
def ca_alpha(n_train: int, pfa: float) -> float:
    """CA-CFAR multiplier ``alpha`` such that ``T = alpha * mean(training)`` gives ``pfa``."""
    n_train = np.asarray(n_train, dtype=float)
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def ca_pfa(alpha: float, n_train: int) -> float:
    return (1.0 + alpha / n_train) ** (-n_train)


def os_pfa(alpha: float, n_train: int, k: int) -> float:
    i = np.arange(k)
    return float(np.prod((n_train - i) / (n_train - i + alpha)))


def os_alpha(n_train: int, k: int, pfa: float) -> float:
    """Solve the OS-CFAR P_FA equation for ``alpha`` (``T = alpha * z_(k)``)."""
    if not 1 <= k <= n_train:
        raise ValueError("k must satisfy 1 <= k <= n_train")
    f = lambda a: np.log(os_pfa(a, n_train, k)) - np.log(pfa)  # noqa: E731
    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
    return float(optimize.brentq(f, 0.0, hi, xtol=1e-14, rtol=1e-14))


# --------------------------------------------------------------------------- 1-D CA-CFAR
def _window_sums_1d(z: np.ndarray, n_train: int, n_guard: int, edge: str):
    """Sum and count of training cells for every CUT along the last axis."""
    n = z.shape[-1]
    reach = n_guard + n_train
    if edge == "wrap":
        zp = np.concatenate([z[..., -reach:], z, z[..., :reach]], axis=-1)
        valid = np.ones_like(zp, dtype=float)
    else:
        pad = [(0, 0)] * (z.ndim - 1) + [(reach, reach)]
        zp = np.pad(z, pad)
        valid = np.pad(np.ones(z.shape, dtype=float), pad)
    cs = np.concatenate([np.zeros(zp.shape[:-1] + (1,)), np.cumsum(zp, axis=-1)], axis=-1)
    cv = np.concatenate([np.zeros(valid.shape[:-1] + (1,)), np.cumsum(valid, axis=-1)], axis=-1)
    idx = np.arange(n) + reach  # CUT index in padded array
    # leading: [idx - reach, idx - n_guard), lagging: (idx + n_guard, idx + reach]
    lead = cs[..., idx - n_guard] - cs[..., idx - reach]
    lag = cs[..., idx + reach + 1] - cs[..., idx + n_guard + 1]
    lead_c = cv[..., idx - n_guard] - cv[..., idx - reach]
    lag_c = cv[..., idx + reach + 1] - cv[..., idx + n_guard + 1]
    return lead + lag, lead_c + lag_c


def ca_cfar_1d(z: np.ndarray, n_train: int, n_guard: int, pfa: float, edge: str = "valid"):
    """CA-CFAR along the last axis of power data ``z``.

    Parameters
    ----------
    z : ndarray
        Square-law samples ``|x|^2``.
    n_train, n_guard : int
        Training and guard cells **per side**. Total training cells ``N_t = 2 n_train``.
    pfa : float
        Design false-alarm probability per cell.
    edge : {"valid", "truncate", "wrap"}

    Returns
    -------
    detections : bool ndarray (same shape as ``z``)
    threshold : float ndarray (NaN where no decision is made)
    """
    if edge not in ("valid", "truncate", "wrap"):
        raise ValueError("edge must be 'valid', 'truncate' or 'wrap'")
    z = np.asarray(z, dtype=float)
    s, c = _window_sums_1d(z, n_train, n_guard, edge)
    full = 2 * n_train
    if edge == "valid":
        ok = c == full
    else:
        ok = c >= 1
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.where(ok, ca_alpha(np.maximum(c, 1), pfa), np.nan)
        thr = np.where(ok, alpha * s / np.maximum(c, 1), np.nan)
    det = np.zeros(z.shape, dtype=bool)
    det[ok] = z[ok] > thr[ok]
    return det, thr


# --------------------------------------------------------------------------- 2-D CA-CFAR
def _box_sum(a: np.ndarray, hk: int, hl: int) -> np.ndarray:
    """Sum over the (2hk+1) x (2hl+1) box centred on each interior cell of a padded array."""
    cs = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    K, L = a.shape
    k = np.arange(hk, K - hk)[:, None]
    l = np.arange(hl, L - hl)[None, :]
    return (cs[k + hk + 1, l + hl + 1] - cs[k - hk, l + hl + 1]
            - cs[k + hk + 1, l - hl] + cs[k - hk, l - hl])


def ca_cfar_2d(
    z: np.ndarray,
    n_train: tuple[int, int],
    n_guard: tuple[int, int],
    pfa: float,
    wrap_axis0: bool = True,
    edge: str = "valid",
):
    """2-D CA-CFAR on a range-Doppler power map ``z`` of shape ``(K_doppler, N_range)``.

    The window is a rectangle of half-size ``(n_guard + n_train)`` per axis with the
    guard rectangle (half-size ``n_guard``, including the CUT) removed. Axis 0
    (Doppler) is treated circularly if ``wrap_axis0``; axis 1 (range) uses ``edge``
    ("valid" or "truncate").
    """
    if edge not in ("valid", "truncate"):
        raise ValueError("edge must be 'valid' or 'truncate' for the range axis")
    z = np.asarray(z, dtype=float)
    tk, tl = n_train
    gk, gl = n_guard
    rk, rl = tk + gk, tl + gl
    valid = np.ones_like(z)
    if wrap_axis0:
        zp = np.pad(z, ((rk, rk), (0, 0)), mode="wrap")
        vp = np.pad(valid, ((rk, rk), (0, 0)), mode="wrap")
    else:
        zp = np.pad(z, ((rk, rk), (0, 0)))
        vp = np.pad(valid, ((rk, rk), (0, 0)))
    zp = np.pad(zp, ((0, 0), (rl, rl)))
    vp = np.pad(vp, ((0, 0), (rl, rl)))
    outer = _box_sum(zp, rk, rl)
    outer_c = _box_sum(vp, rk, rl)
    # guard box: crop the padded arrays so the guard box is centred on the same cells
    zg = zp[rk - gk: zp.shape[0] - (rk - gk), rl - gl: zp.shape[1] - (rl - gl)]
    vg = vp[rk - gk: vp.shape[0] - (rk - gk), rl - gl: vp.shape[1] - (rl - gl)]
    inner = _box_sum(zg, gk, gl)
    inner_c = _box_sum(vg, gk, gl)
    s = outer - inner
    c = outer_c - inner_c
    full = (2 * rk + 1) * (2 * rl + 1) - (2 * gk + 1) * (2 * gl + 1)
    ok = (np.round(c) == full) if edge == "valid" else (c >= 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.maximum(np.round(c), 1)
        thr = np.where(ok, ca_alpha(cc, pfa) * s / cc, np.nan)
    det = np.zeros(z.shape, dtype=bool)
    det[ok] = z[ok] > thr[ok]
    return det, thr, int(full)


# --------------------------------------------------------------------------- OS-CFAR (1-D)
def os_cfar_1d(z: np.ndarray, n_train: int, n_guard: int, k: int, pfa: float):
    """OS-CFAR along the last axis, ``edge="valid"`` only. ``k`` is 1-based (k-th smallest)."""
    from numpy.lib.stride_tricks import sliding_window_view

    z = np.asarray(z, dtype=float)
    n = z.shape[-1]
    reach = n_guard + n_train
    win = sliding_window_view(z, 2 * reach + 1, axis=-1)  # (..., n - 2 reach, 2 reach + 1)
    train = np.concatenate([win[..., :n_train], win[..., -n_train:]], axis=-1)
    zk = np.partition(train, k - 1, axis=-1)[..., k - 1]
    alpha = os_alpha(2 * n_train, k, pfa)
    thr = np.full(z.shape, np.nan)
    thr[..., reach:n - reach] = alpha * zk
    det = np.zeros(z.shape, dtype=bool)
    det[..., reach:n - reach] = z[..., reach:n - reach] > thr[..., reach:n - reach]
    return det, thr


# --------------------------------------------------------------------------- detection probability
def pd_fixed_threshold_swerling0(snr: float, pfa: float) -> float:
    """Known-noise (ideal) detector, non-fluctuating target: Marcum-Q ``Q1(sqrt(2 snr), sqrt(-2 ln pfa))``."""
    t = -np.log(pfa)
    return float(stats.ncx2.sf(2.0 * t, 2, 2.0 * snr))


def pd_fixed_threshold_swerling1(snr: float, pfa: float) -> float:
    """Known-noise detector, Rayleigh-fluctuating target: ``pfa^(1/(1+snr))``."""
    return float(pfa ** (1.0 / (1.0 + snr)))


def pd_ca_swerling1(snr: float, pfa: float, n_train: int) -> float:
    """CA-CFAR, Swerling-1 target: ``(1 + alpha / (N_t (1 + snr)))^(-N_t)``."""
    a = ca_alpha(n_train, pfa)
    return float((1.0 + a / (n_train * (1.0 + snr))) ** (-n_train))


def pd_ca_swerling0(snr: float, pfa: float, n_train: int) -> float:
    """CA-CFAR, non-fluctuating target, by quadrature over the Gamma-distributed noise estimate.

    With unit noise power the threshold is ``T = (alpha/N_t) Z``, ``Z ~ Gamma(N_t, 1)``, and
    ``P(|A + n|^2 > T) = ncx2.sf(2T; 2, 2 snr)``.
    """
    a = ca_alpha(n_train, pfa)
    g = stats.gamma(n_train)
    f = lambda zz: stats.ncx2.sf(2.0 * a * zz / n_train, 2, 2.0 * snr) * g.pdf(zz)  # noqa: E731
    lo, hi = g.ppf(1e-14), g.ppf(1 - 1e-14)
    val, _ = integrate.quad(f, lo, hi, limit=400, epsabs=1e-12)
    return float(val)


def required_snr_db(pd_target: float, pd_func, *args) -> float:
    """SNR (dB) at which ``pd_func(snr, *args) = pd_target`` (root-find in dB)."""
    g = lambda s_db: pd_func(10 ** (s_db / 10), *args) - pd_target  # noqa: E731
    return float(optimize.brentq(g, -20.0, 60.0, xtol=1e-6))


def binomial_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval for ``k`` successes in ``n`` trials."""
    a = 1 - conf
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)

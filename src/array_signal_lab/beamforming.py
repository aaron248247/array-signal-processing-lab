"""Narrowband beamformers.

The beamformer output for snapshot ``x`` is ``y = w^H x``. The response of
weight vector ``w`` to a unit plane wave from ``theta`` is ``w^H a(theta)``.
"""

from __future__ import annotations

import numpy as np

from .array_model import ULA
from .steering import steering_vector


def conventional_weights(ula: ULA, theta0: float) -> np.ndarray:
    """Delay-and-sum weights ``w = a(theta0) / N``.

    Normalised so that ``w^H a(theta0) = 1`` (unit gain toward the look
    direction). Then ``||w||^2 = 1/N`` and the white-noise array gain is ``N``.
    """
    return steering_vector(ula, theta0) / ula.n_elements


def response(w: np.ndarray, ula: ULA, theta) -> np.ndarray:
    """Complex response ``w^H a(theta)`` (the numerical array factor)."""
    return np.conj(w) @ steering_vector(ula, theta)


def beampattern_db(w: np.ndarray, ula: ULA, theta, normalize: bool = True) -> np.ndarray:
    """Power pattern ``20 log10 |w^H a(theta)|`` in dB.

    If ``normalize`` the pattern is referenced to its maximum over ``theta``.
    """
    mag = np.abs(response(w, ula, theta))
    db = 20.0 * np.log10(np.maximum(mag, 1e-300))
    if normalize:
        db = db - db.max()
    return db


def angle_scan(ula: ULA, snapshots: np.ndarray, thetas) -> np.ndarray:
    """Conventional (Bartlett) spatial spectrum: mean output power when steering over ``thetas``.

    ``P(theta) = (1/K) sum_k |w(theta)^H x_k|^2 = w^H R_hat w`` with ``w = a/N``.

    Parameters
    ----------
    snapshots : ndarray, shape ``(N, K)``
    thetas : array_like, shape ``(M,)``

    Returns
    -------
    ndarray, shape ``(M,)``
    """
    W = steering_vector(ula, thetas) / ula.n_elements  # (N, M)
    y = np.conj(W).T @ snapshots  # (M, K)
    return np.mean(np.abs(y) ** 2, axis=1)


class SingularCovarianceError(np.linalg.LinAlgError):
    """Raised when the (loaded) covariance is singular to working precision."""


def mvdr_weights(
    R: np.ndarray,
    a0: np.ndarray,
    loading: float = 0.0,
    rcond_min: float = 1e-12,
) -> np.ndarray:
    """MVDR / Capon weights ``w = R_l^{-1} a0 / (a0^H R_l^{-1} a0)``, ``R_l = R + loading I``.

    Minimises ``w^H R w`` subject to ``w^H a0 = 1`` [Capon 1969].

    Implementation: Cholesky factorisation of ``R_l`` and two triangular solves
    (no explicit inverse). If ``R_l`` is not positive definite, or its reciprocal
    condition number ``lambda_min / lambda_max`` is below ``rcond_min``, a
    :class:`SingularCovarianceError` is raised instead of returning weights
    dominated by round-off. This is the expected outcome for an unloaded sample
    covariance with fewer snapshots than elements (rank ``K < N``).
    """
    from scipy.linalg import cho_factor, cho_solve

    n = R.shape[0]
    Rl = R + loading * np.eye(n, dtype=complex) if loading else np.asarray(R, dtype=complex)
    Rl = 0.5 * (Rl + Rl.conj().T)
    ev = np.linalg.eigvalsh(Rl)
    if ev[0] <= 0 or ev[0] / ev[-1] < rcond_min:
        raise SingularCovarianceError(
            f"covariance singular to working precision (lambda_min/lambda_max = {ev[0] / ev[-1]:.3e})"
        )
    c = cho_factor(Rl, lower=True)
    u = cho_solve(c, a0)
    return u / np.vdot(a0, u)

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

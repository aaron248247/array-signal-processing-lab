"""Snapshot simulation and covariance estimation.

Data model
----------
Narrowband snapshots, ``k = 1..K``:

    x_k = sum_i s_i[k] a(theta_i) + n_k,

with ``s_i[k] ~ CN(0, p_i)`` mutually independent and independent over ``k``,
and ``n_k ~ CN(0, sigma^2 I)``. The ensemble covariance is

    R = E[x x^H] = sum_i p_i a(theta_i) a(theta_i)^H + sigma^2 I.

The sample covariance is ``R_hat = (1/K) X X^H`` with ``X = [x_1 ... x_K]``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .array_model import ULA
from .steering import steering_vector


def complex_gaussian(rng: np.random.Generator, shape, power: float = 1.0) -> np.ndarray:
    """Circularly-symmetric complex Gaussian samples with ``E|z|^2 = power``."""
    scale = np.sqrt(power / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def simulate_snapshots(
    ula: ULA,
    angles: Sequence[float],
    powers: Sequence[float],
    noise_power: float,
    n_snapshots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``K`` snapshots from the narrowband model. Returns ``X`` of shape ``(N, K)``."""
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    powers = np.atleast_1d(np.asarray(powers, dtype=float))
    if angles.shape != powers.shape:
        raise ValueError("angles and powers must have the same length")
    n = ula.n_elements
    X = complex_gaussian(rng, (n, n_snapshots), noise_power)
    if angles.size:
        A = steering_vector(ula, angles).reshape(n, -1)  # (N, M)
        S = complex_gaussian(rng, (angles.size, n_snapshots), 1.0) * np.sqrt(powers)[:, None]
        X = X + A @ S
    return X


def theoretical_covariance(
    ula: ULA, angles: Sequence[float], powers: Sequence[float], noise_power: float
) -> np.ndarray:
    """Ensemble covariance ``sum_i p_i a_i a_i^H + sigma^2 I``."""
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    powers = np.atleast_1d(np.asarray(powers, dtype=float))
    n = ula.n_elements
    R = noise_power * np.eye(n, dtype=complex)
    if angles.size:
        A = steering_vector(ula, angles).reshape(n, -1)
        R = R + (A * powers) @ A.conj().T
    return R


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """Sample covariance ``(1/K) X X^H``, symmetrised to be exactly Hermitian."""
    if X.ndim != 2:
        raise ValueError("X must have shape (N, K)")
    k = X.shape[1]
    R = (X @ X.conj().T) / k
    return 0.5 * (R + R.conj().T)


def diagonal_load(R: np.ndarray, loading: float) -> np.ndarray:
    """Return ``R + loading * I``."""
    return R + loading * np.eye(R.shape[0], dtype=R.dtype)


def condition_number(R: np.ndarray) -> float:
    """2-norm condition number of a Hermitian matrix from its eigenvalues (``inf`` if singular)."""
    ev = np.linalg.eigvalsh(R)
    lo, hi = ev.min(), ev.max()
    if lo <= 0:
        return float("inf")
    return float(hi / lo)

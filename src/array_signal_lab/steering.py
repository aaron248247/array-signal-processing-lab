"""Steering vectors and closed-form ULA array factor.

Steering vector
---------------
For a plane wave from angle ``theta`` (measured from broadside)

    a_n(theta) = exp(j k x_n sin(theta)),   x_n = n d,   n = 0..N-1.

``a`` is *not* normalized: ``a^H a = N``.

Array factor
------------
For weights ``w`` the array response is ``y(theta) = w^H a(theta)``. With the
conventional (delay-and-sum) weights ``w = a(theta_0) / N`` the sum is a
geometric series in ``exp(j psi)`` with

    psi = k d (sin(theta) - sin(theta_0)),

giving

    AF(theta) = (1/N) sum_n exp(j n psi)
              = exp(j (N-1) psi / 2) * sin(N psi / 2) / (N sin(psi / 2)).

The phase factor reflects the choice of element 0 as phase reference.
"""

from __future__ import annotations

import numpy as np

from .array_model import ULA


def steering_vector(ula: ULA, theta) -> np.ndarray:
    """Steering vector(s) ``a(theta)``.

    Parameters
    ----------
    ula : ULA
    theta : float or array_like
        Angle(s) from broadside in radians.

    Returns
    -------
    ndarray
        Shape ``(N,)`` for scalar ``theta``; ``(N, M)`` for ``M`` angles.
    """
    theta = np.asarray(theta, dtype=float)
    phase = ula.wavenumber * np.multiply.outer(ula.positions, np.sin(theta))
    return np.exp(1j * phase)


def spatial_phase(ula: ULA, theta, theta0: float = 0.0) -> np.ndarray:
    """Inter-element phase difference ``psi = k d (sin theta - sin theta0)``."""
    theta = np.asarray(theta, dtype=float)
    return ula.wavenumber * ula.spacing * (np.sin(theta) - np.sin(theta0))


def array_factor_analytical(ula: ULA, theta, theta0: float = 0.0) -> np.ndarray:
    """Closed-form complex array factor of the uniformly weighted ULA steered to ``theta0``.

    Normalized so that ``AF(theta0) = 1``. The removable singularity at
    ``sin(psi/2) = 0`` (main lobe and grating lobes) is evaluated as its limit,
    ``exp(j (N-1) psi / 2) * (+/-1)``.
    """
    n = ula.n_elements
    psi = spatial_phase(ula, theta, theta0)
    half = psi / 2.0
    num = np.sin(n * half)
    den = n * np.sin(half)
    # Limit at sin(psi/2) -> 0: psi/2 = m pi  ->  ratio -> cos(N m pi)/cos(m pi) = (+/-1)
    small = np.abs(den) < 1e-12
    ratio = np.empty_like(np.asarray(psi, dtype=float))
    ratio[~small] = num[~small] / den[~small]
    m = np.round(half[small] / np.pi)
    ratio[small] = np.cos(n * m * np.pi) / np.cos(m * np.pi)
    out = np.exp(1j * (n - 1) * half) * ratio
    return out if out.ndim else out[()]


def null_angles(ula: ULA, theta0: float = 0.0) -> np.ndarray:
    """Analytical null directions of the uniformly weighted ULA steered to ``theta0``.

    Nulls occur where ``N psi / 2 = m pi`` with ``m`` not a multiple of ``N``,
    i.e. ``sin(theta) = sin(theta0) + m lambda / (N d)``. Only nulls in visible
    space (``|sin theta| <= 1``) are returned, sorted ascending.
    """
    n = ula.n_elements
    step = 1.0 / (n * ula.spacing_wavelengths)
    m_max = int(np.ceil(2.0 / step)) + 1
    m = np.arange(-m_max, m_max + 1)
    m = m[(m % n) != 0]
    s = np.sin(theta0) + m * step
    s = s[np.abs(s) <= 1.0]
    return np.sort(np.arcsin(s))


def first_null_beamwidth(ula: ULA, theta0: float = 0.0) -> float:
    """Angular distance (rad) between the two first nulls around ``theta0``.

    Returns ``nan`` if either first null is outside visible space.
    """
    step = 1.0 / (ula.n_elements * ula.spacing_wavelengths)
    s0 = np.sin(theta0)
    lo, hi = s0 - step, s0 + step
    if abs(lo) > 1 or abs(hi) > 1:
        return float("nan")
    return float(np.arcsin(hi) - np.arcsin(lo))

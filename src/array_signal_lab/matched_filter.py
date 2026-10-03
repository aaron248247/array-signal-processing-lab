"""Matched filtering (pulse compression) and peak interpolation.

The matched filter for replica ``h[n]`` (length ``L``) is the correlation

    y[l] = sum_n x[l + n] conj(h[n]) / ||h||,      l = 0 .. N_s - 1,

so that a delayed copy ``x[n] = A h[n - l0]`` produces a peak ``A ||h||`` at
lag ``l0`` and white noise of variance ``sigma^2`` keeps variance ``sigma^2``.
The output SNR gain for a unit-modulus replica is therefore ``L``.

The FFT implementation zero-pads to at least ``N_s + L - 1`` so that the
circular correlation equals the linear one for all lags ``0..N_s-1``.
Lags near the end of the window (``l > N_s - L``) see only a partial echo.
"""

from __future__ import annotations

import numpy as np


def _nfft(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 1))))


def matched_filter(x: np.ndarray, replica: np.ndarray, axis: int = -1) -> np.ndarray:
    """FFT-based matched filter along ``axis``. Output has the same shape as ``x``."""
    x = np.asarray(x)
    h = np.asarray(replica, dtype=complex)
    ns = x.shape[axis]
    nfft = _nfft(ns + h.size - 1)
    X = np.fft.fft(x, nfft, axis=axis)
    H = np.fft.fft(h, nfft)
    shape = [1] * x.ndim
    shape[axis] = nfft
    y = np.fft.ifft(X * np.conj(H).reshape(shape), axis=axis)
    y = np.take(y, np.arange(ns), axis=axis)
    return y / np.linalg.norm(h)


def matched_filter_direct(x: np.ndarray, replica: np.ndarray) -> np.ndarray:
    """Time-domain reference implementation for a 1-D input (used for testing)."""
    x = np.asarray(x, dtype=complex)
    h = np.asarray(replica, dtype=complex)
    ns, L = x.size, h.size
    xp = np.concatenate([x, np.zeros(L, dtype=complex)])
    y = np.array([np.sum(xp[l:l + L] * np.conj(h)) for l in range(ns)])
    return y / np.linalg.norm(h)


def parabolic_peak(values: np.ndarray, index: int) -> float:
    """Sub-sample peak offset in ``(-0.5, 0.5)`` from a parabola through three samples.

    ``values`` should be a magnitude (``|y|``); returns 0 at array edges.
    """
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    a, b, c = values[index - 1], values[index], values[index + 1]
    den = a - 2.0 * b + c
    if den == 0:
        return 0.0
    return float(0.5 * (a - c) / den)


def estimate_delay_samples(y: np.ndarray, interpolate: bool = True, search: slice | None = None) -> float:
    """Fractional-sample location of the largest ``|y|`` (optionally within ``search``)."""
    mag = np.abs(y)
    offset = 0
    if search is not None:
        offset = search.start or 0
        mag_s = mag[search]
    else:
        mag_s = mag
    i = int(np.argmax(mag_s)) + offset
    return i + (parabolic_peak(mag, i) if interpolate else 0.0)

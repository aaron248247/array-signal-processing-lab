"""Pulse-to-pulse (slow-time) Doppler processing and range-Doppler maps.

For each range cell the ``M`` slow-time samples are windowed and transformed:

    Y[k, l] = sum_m g[m] y_m[l] exp(-j 2 pi k m / K),   k = 0..K-1  (K >= M, zero-padded),

then ``fftshift``-ed so the Doppler axis runs ``[-PRF/2, PRF/2)``. With the
``exp(+j 2 pi f_D t_m)`` echo convention of :mod:`radar_signal`, a closing
target (``f_D > 0``) appears at a positive Doppler frequency.

Window normalisation: ``g`` is scaled to ``sum g^2 = 1`` so white noise keeps
its per-cell variance; the coherent gain for a bin-centred target is then
``(sum g)^2 / sum g^2`` (``= M`` for the rectangular window).
"""

from __future__ import annotations

import numpy as np


def window(name: str, m: int) -> np.ndarray:
    if name in ("rect", "rectangular", "none"):
        g = np.ones(m)
    elif name in ("hann", "hanning"):
        g = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(m) / m)  # periodic ("DFT-even") Hann [Harris 1978]
    elif name == "hamming":
        g = np.hamming(m)
    else:
        raise ValueError(f"unknown window {name!r}")
    return g / np.sqrt(np.sum(g**2))


def coherent_gain(name: str, m: int) -> float:
    g = window(name, m)
    return float(np.sum(g) ** 2 / np.sum(g**2))


def doppler_process(data: np.ndarray, win: str = "rect", nfft: int | None = None) -> np.ndarray:
    """Slow-time FFT along axis 0 of ``data`` (shape ``(M, N_s)``). Returns ``(K, N_s)``, fftshift-ed."""
    m = data.shape[0]
    k = m if nfft is None else int(nfft)
    if k < m:
        raise ValueError("nfft must be >= number of pulses")
    g = window(win, m)[:, None]
    Y = np.fft.fft(data * g, n=k, axis=0)
    return np.fft.fftshift(Y, axes=0)


def doppler_axis(prf_hz: float, nfft: int) -> np.ndarray:
    """Doppler frequencies (Hz) matching :func:`doppler_process` output ordering."""
    return np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / prf_hz))


def scalloping_loss_db(win: str, m: int, offset_bins: float) -> float:
    """Exact loss (dB, negative) of a tone ``offset_bins`` from a bin centre relative to bin-centred."""
    g = window(win, m)
    n = np.arange(m)
    centred = np.abs(np.sum(g))
    off = np.abs(np.sum(g * np.exp(2j * np.pi * offset_bins * n / m)))
    return float(20 * np.log10(off / centred))

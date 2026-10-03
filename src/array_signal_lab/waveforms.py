"""Transmit waveforms (complex baseband).

Linear FM (LFM) pulse of duration ``T`` and swept bandwidth ``B``:

    u(t) = exp(j pi mu (t - T/2)^2),   0 <= t < T,   mu = B / T,

so the instantaneous frequency ``mu (t - T/2)`` sweeps ``-B/2 -> +B/2``.

Closed-form zero-Doppler autocorrelation magnitude (continuous time)
[Levanon & Mozeson 2004, ch. 4]::

    |chi(tau, 0)| / T = |(1 - |tau|/T) sinc(B tau (1 - |tau|/T))|,  |tau| < T,

with ``sinc(x) = sin(pi x)/(pi x)``. The -4 dB mainlobe width is ``~1/B``,
giving the range resolution ``c / (2B)``.

Range-Doppler coupling: a Doppler shift ``f_D`` moves the matched-filter peak
of an up-chirp to ``tau = -f_D / mu = -f_D T / B``.
"""

from __future__ import annotations

import numpy as np


def lfm(t, bandwidth: float, duration: float) -> np.ndarray:
    """Evaluate the LFM pulse at arbitrary times ``t`` (zero outside ``[0, T)``)."""
    t = np.asarray(t, dtype=float)
    mu = bandwidth / duration
    inside = (t >= 0.0) & (t < duration)
    out = np.zeros(t.shape, dtype=complex)
    tc = t[inside] - duration / 2.0
    out[inside] = np.exp(1j * np.pi * mu * tc**2)
    return out


def lfm_pulse(bandwidth: float, duration: float, fs: float) -> np.ndarray:
    """Sampled LFM pulse, ``n = 0 .. round(T fs) - 1``, unit-modulus samples."""
    n = int(round(duration * fs))
    return lfm(np.arange(n) / fs, bandwidth, duration)


def rect_pulse(duration: float, fs: float) -> np.ndarray:
    """Sampled unmodulated rectangular pulse (unit-modulus samples)."""
    return np.ones(int(round(duration * fs)), dtype=complex)


def lfm_autocorrelation_magnitude(tau, bandwidth: float, duration: float) -> np.ndarray:
    """Normalized ``|chi(tau, 0)| / T`` of the continuous LFM pulse (peak = 1)."""
    tau = np.asarray(tau, dtype=float)
    r = 1.0 - np.abs(tau) / duration
    out = np.where(np.abs(tau) < duration, np.abs(r * np.sinc(bandwidth * tau * r)), 0.0)
    return out


def lfm_coupling_delay(doppler_hz: float, bandwidth: float, duration: float) -> float:
    """Delay offset of the matched-filter peak caused by a Doppler shift: ``-f_D T / B``."""
    return -doppler_hz * duration / bandwidth

"""Generic monostatic pulse-Doppler signal model (complex baseband I/Q).

Conventions
-----------
* Radial velocity ``v > 0`` means the target is **closing** (range decreasing):
  ``R(t) = R0 - v t``.
* Round-trip delay ``tau = 2R/c``; Doppler shift ``f_D = 2 v / lambda`` (positive
  for a closing target).
* Fast time ``t_n = n / f_s`` is measured from the start of each transmitted
  pulse, ``n = 0..N_s-1``; slow time ``t_m = m T_r``, ``m = 0..M-1``,
  ``T_r = 1/PRF``.

Echo of pulse ``m`` (narrowband, "stop-and-hop" for the envelope within one
pulse; carrier phase evaluated continuously):

    x_m[n] = A u(t_n - tau_m) exp(j 2 pi f_D (t_m + t_n)) exp(-j 4 pi R0 / lambda) + w_m[n],
    tau_m  = 2 (R0 - v t_m) / c,

where ``u`` is the transmit envelope and ``w ~ CN(0, sigma^2)`` is white.
Range migration of the envelope across the CPI is included through ``tau_m``;
the time-compression of the envelope by ``(1 - 2v/c)`` is neglected
(relative error ``2v/c`` ~ 1e-6 for the velocities used here).

All parameter values in this repository are generic round numbers chosen for
illustration; they do not describe any particular radar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .covariance import complex_gaussian
from .waveforms import lfm

C0 = 299_792_458.0  # m/s


@dataclass(frozen=True)
class RadarParams:
    carrier_hz: float = 3.0e9
    bandwidth_hz: float = 5.0e6
    pulse_s: float = 10.0e-6
    fs_hz: float = 10.0e6
    prf_hz: float = 10.0e3
    n_pulses: int = 64
    c: float = C0

    @property
    def wavelength(self) -> float:
        return self.c / self.carrier_hz

    @property
    def pri(self) -> float:
        return 1.0 / self.prf_hz

    @property
    def n_fast(self) -> int:
        """Fast-time samples per PRI (receive window = full PRI)."""
        return int(round(self.pri * self.fs_hz))

    @property
    def n_pulse_samples(self) -> int:
        return int(round(self.pulse_s * self.fs_hz))

    @property
    def range_bin(self) -> float:
        """Range-sample spacing ``c / (2 f_s)``."""
        return self.c / (2.0 * self.fs_hz)

    @property
    def range_resolution(self) -> float:
        """Nominal LFM range resolution ``c / (2B)``."""
        return self.c / (2.0 * self.bandwidth_hz)

    @property
    def unambiguous_range(self) -> float:
        return self.c / (2.0 * self.prf_hz)

    @property
    def doppler_bin(self) -> float:
        """Doppler FFT bin spacing ``PRF / M`` (Hz)."""
        return self.prf_hz / self.n_pulses

    @property
    def velocity_bin(self) -> float:
        """Velocity bin spacing ``lambda PRF / (2M)``."""
        return self.wavelength * self.doppler_bin / 2.0

    @property
    def unambiguous_velocity(self) -> float:
        """Velocities are unambiguous in ``[-v_u, v_u)`` with ``v_u = lambda PRF / 4``."""
        return self.wavelength * self.prf_hz / 4.0

    @property
    def time_bandwidth(self) -> float:
        return self.bandwidth_hz * self.pulse_s

    def doppler_hz(self, velocity: float) -> float:
        return 2.0 * velocity / self.wavelength

    def range_axis(self) -> np.ndarray:
        """Range of each fast-time lag after matched filtering (lag 0 = zero delay)."""
        return np.arange(self.n_fast) * self.range_bin


@dataclass(frozen=True)
class PointTarget:
    range_m: float
    velocity_mps: float
    snr_db: float = 0.0  # per-sample SNR of the raw echo (before any processing)
    phase_rad: float = 0.0


def simulate_cpi(
    params: RadarParams,
    targets: Sequence[PointTarget],
    noise_power: float = 1.0,
    rng: np.random.Generator | None = None,
    include_range_migration: bool = True,
) -> np.ndarray:
    """Simulate one coherent processing interval. Returns ``(M, N_s)`` complex samples.

    If ``rng`` is None the data are noiseless.
    """
    M, Ns = params.n_pulses, params.n_fast
    t_fast = np.arange(Ns) / params.fs_hz
    t_slow = np.arange(M) * params.pri
    X = np.zeros((M, Ns), dtype=complex)
    lam = params.wavelength
    for tg in targets:
        amp = np.sqrt(noise_power * 10 ** (tg.snr_db / 10)) * np.exp(1j * tg.phase_rad)
        fd = params.doppler_hz(tg.velocity_mps)
        if include_range_migration:
            tau = 2.0 * (tg.range_m - tg.velocity_mps * t_slow) / params.c
        else:
            tau = np.full(M, 2.0 * tg.range_m / params.c)
        env = lfm(t_fast[None, :] - tau[:, None], params.bandwidth_hz, params.pulse_s)
        phase = np.exp(1j * 2 * np.pi * fd * (t_slow[:, None] + t_fast[None, :]))
        X += amp * np.exp(-1j * 4 * np.pi * tg.range_m / lam) * env * phase
    if rng is not None:
        X += complex_gaussian(rng, (M, Ns), noise_power)
    return X

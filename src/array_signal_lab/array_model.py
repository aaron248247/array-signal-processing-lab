"""Uniform linear array (ULA) geometry.

Conventions (used everywhere in the package)
--------------------------------------------
* Elements lie on the x axis at ``x_n = n d``, ``n = 0, ..., N-1``.
* ``theta`` is measured from broadside (the array normal), in radians,
  ``theta in [-pi/2, pi/2]``. Positive ``theta`` tilts toward +x.
* A far-field narrowband plane wave arriving from ``theta`` produces the
  element phase ``exp(+j k x_n sin(theta))`` relative to element 0, with
  ``k = 2 pi / lambda`` (see :mod:`array_signal_lab.steering`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ULA:
    """Narrowband uniform linear array.

    Parameters
    ----------
    n_elements : int
        Number of elements ``N``.
    spacing_wavelengths : float
        Element spacing ``d / lambda``. Default 0.5 (half-wavelength).
    wavelength : float
        Carrier wavelength ``lambda`` in metres. Only the ratio ``d/lambda``
        enters the narrowband steering vector; the absolute value is kept so
        that positions can be reported in metres.
    """

    n_elements: int = 16
    spacing_wavelengths: float = 0.5
    wavelength: float = 1.0

    def __post_init__(self) -> None:
        if int(self.n_elements) != self.n_elements or self.n_elements < 1:
            raise ValueError("n_elements must be a positive integer")
        if self.spacing_wavelengths <= 0:
            raise ValueError("spacing_wavelengths must be positive")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")

    @property
    def spacing(self) -> float:
        """Element spacing ``d`` in metres."""
        return self.spacing_wavelengths * self.wavelength

    @property
    def wavenumber(self) -> float:
        """Free-space wavenumber ``k = 2 pi / lambda`` in rad/m."""
        return 2.0 * np.pi / self.wavelength

    @property
    def positions(self) -> np.ndarray:
        """Element x-coordinates ``x_n = n d`` in metres, shape ``(N,)``."""
        return element_positions(self.n_elements, self.spacing)

    @property
    def aperture(self) -> float:
        """Physical aperture ``(N-1) d`` in metres."""
        return (self.n_elements - 1) * self.spacing

    def has_grating_lobes(self, scan_max: float = np.pi / 2) -> bool:
        """True if a grating lobe can enter visible space for scans up to ``scan_max``.

        A grating-lobe peak lies strictly inside visible space when
        ``d/lambda > 1 / (1 + |sin(scan_max)|)``. At equality it sits at endfire.
        """
        return bool(self.spacing_wavelengths > 1.0 / (1.0 + abs(np.sin(scan_max))))


def element_positions(n_elements: int, spacing: float) -> np.ndarray:
    """Return ``x_n = n d`` for ``n = 0..N-1``."""
    return np.arange(n_elements, dtype=float) * spacing

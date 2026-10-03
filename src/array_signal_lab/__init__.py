"""Array signal processing, adaptive beamforming, pulse-Doppler processing and CFAR detection.

Angle convention: ``theta`` from broadside, radians. Steering vector
``a_n(theta) = exp(j k n d sin theta)``. Beamformer output ``y = w^H x``.
"""

from .array_model import ULA, element_positions
from .steering import (
    array_factor_analytical,
    first_null_beamwidth,
    null_angles,
    spatial_phase,
    steering_vector,
)
from .beamforming import angle_scan, beampattern_db, conventional_weights, response

__all__ = [
    "ULA",
    "element_positions",
    "steering_vector",
    "spatial_phase",
    "array_factor_analytical",
    "null_angles",
    "first_null_beamwidth",
    "conventional_weights",
    "response",
    "beampattern_db",
    "angle_scan",
]

__version__ = "0.1.0"

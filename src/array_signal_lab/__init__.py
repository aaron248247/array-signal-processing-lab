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
from .beamforming import (
    SingularCovarianceError,
    angle_scan,
    beampattern_db,
    conventional_weights,
    mvdr_weights,
    response,
)
from .covariance import (
    complex_gaussian,
    condition_number,
    diagonal_load,
    sample_covariance,
    simulate_snapshots,
    theoretical_covariance,
)

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
    "mvdr_weights",
    "SingularCovarianceError",
    "complex_gaussian",
    "simulate_snapshots",
    "theoretical_covariance",
    "sample_covariance",
    "diagonal_load",
    "condition_number",
]

__version__ = "0.1.0"

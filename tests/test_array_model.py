import numpy as np
import pytest

from array_signal_lab import (
    ULA,
    angle_scan,
    array_factor_analytical,
    conventional_weights,
    first_null_beamwidth,
    null_angles,
    response,
    steering_vector,
)


@pytest.fixture(params=[8, 16])
def ula(request):
    return ULA(n_elements=request.param, spacing_wavelengths=0.5, wavelength=0.1)


def test_positions_and_spacing(ula):
    x = ula.positions
    assert x.shape == (ula.n_elements,)
    assert x[0] == 0.0
    np.testing.assert_allclose(np.diff(x), ula.spacing)
    assert ula.spacing == pytest.approx(0.05)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ULA(n_elements=0)
    with pytest.raises(ValueError):
        ULA(spacing_wavelengths=-0.5)


def test_steering_vector_shape(ula):
    assert steering_vector(ula, 0.1).shape == (ula.n_elements,)
    assert steering_vector(ula, np.linspace(-1, 1, 7)).shape == (ula.n_elements, 7)


def test_steering_vector_unit_modulus_and_norm(ula):
    a = steering_vector(ula, np.deg2rad(23.0))
    np.testing.assert_allclose(np.abs(a), 1.0, atol=1e-15)
    assert np.vdot(a, a).real == pytest.approx(ula.n_elements)


def test_phase_progression(ula):
    theta = np.deg2rad(17.0)
    a = steering_vector(ula, theta)
    expected = ula.wavenumber * ula.spacing * np.sin(theta)  # = pi sin(theta) at d = lambda/2
    ratios = a[1:] / a[:-1]
    np.testing.assert_allclose(np.angle(ratios), expected, atol=1e-12)
    assert a[0] == pytest.approx(1.0)


def test_broadside_steering_is_all_ones(ula):
    np.testing.assert_allclose(steering_vector(ula, 0.0), np.ones(ula.n_elements), atol=1e-15)


def test_symmetry_conjugate(ula):
    # a(-theta) = conj(a(theta)) for a real-position ULA
    th = np.deg2rad(31.0)
    np.testing.assert_allclose(steering_vector(ula, -th), np.conj(steering_vector(ula, th)), atol=1e-14)


@pytest.mark.parametrize("theta0_deg", [0.0, 20.0, -35.0, 60.0])
def test_analytical_vs_numerical_array_factor(ula, theta0_deg):
    theta0 = np.deg2rad(theta0_deg)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    w = conventional_weights(ula, theta0)
    numerical = response(w, ula, thetas)
    analytical = array_factor_analytical(ula, thetas, theta0)
    np.testing.assert_allclose(numerical, analytical, atol=1e-12, rtol=0)


@pytest.mark.parametrize("theta0_deg", [0.0, 20.0, -35.0])
def test_main_lobe_direction(ula, theta0_deg):
    theta0 = np.deg2rad(theta0_deg)
    thetas = np.deg2rad(np.linspace(-90, 90, 180001))  # 0.001 deg grid
    w = conventional_weights(ula, theta0)
    peak = thetas[np.argmax(np.abs(response(w, ula, thetas)))]
    assert abs(np.rad2deg(peak) - theta0_deg) <= 0.001 + 1e-9


def test_unit_gain_at_look_direction(ula):
    theta0 = np.deg2rad(12.0)
    w = conventional_weights(ula, theta0)
    assert response(w, ula, theta0) == pytest.approx(1.0 + 0j, abs=1e-14)
    assert np.vdot(w, w).real == pytest.approx(1.0 / ula.n_elements)


@pytest.mark.parametrize("theta0_deg", [0.0, 25.0])
def test_nulls_are_zeros_of_numerical_response(ula, theta0_deg):
    theta0 = np.deg2rad(theta0_deg)
    nulls = null_angles(ula, theta0)
    assert nulls.size > 0
    w = conventional_weights(ula, theta0)
    np.testing.assert_allclose(np.abs(response(w, ula, nulls)), 0.0, atol=1e-12)


def test_broadside_null_count_and_first_null(ula):
    # d = lambda/2, broadside: sin(theta) = m * 2/N, m = +-1..+-(N/2 - 1) -> N-2 visible nulls
    # (m = +-N/2 lands on sin = +-1, endfire, where psi = +-pi and the AF is zero too)
    nulls = null_angles(ula, 0.0)
    n = ula.n_elements
    assert nulls.size == n  # includes the two endfire zeros
    fnbw = first_null_beamwidth(ula, 0.0)
    assert fnbw == pytest.approx(2 * np.arcsin(2.0 / n))


def test_pattern_symmetry_broadside(ula):
    thetas = np.linspace(0, np.pi / 2, 501)
    w = conventional_weights(ula, 0.0)
    np.testing.assert_allclose(np.abs(response(w, ula, thetas)), np.abs(response(w, ula, -thetas)), atol=1e-14)


def test_angle_scan_noiseless_plane_wave_matches_af_squared():
    ula = ULA(16)
    theta_s = np.deg2rad(-12.0)
    rng = np.random.default_rng(1)
    s = (rng.standard_normal(200) + 1j * rng.standard_normal(200)) / np.sqrt(2)
    X = np.outer(steering_vector(ula, theta_s), s)
    thetas = np.deg2rad(np.linspace(-60, 60, 241))
    p = angle_scan(ula, X, thetas)
    expected = np.mean(np.abs(s) ** 2) * np.abs(array_factor_analytical(ula, theta_s, thetas)) ** 2
    np.testing.assert_allclose(p, expected, rtol=1e-10, atol=1e-14)


def test_grating_lobe_flag():
    assert not ULA(16, 0.5).has_grating_lobes()
    assert not ULA(16, 1.0).has_grating_lobes(scan_max=0.0)  # lobe exactly at endfire
    assert ULA(16, 1.0).has_grating_lobes(scan_max=np.deg2rad(5))
    assert ULA(16, 0.6).has_grating_lobes(scan_max=np.deg2rad(60))

import numpy as np
import pytest

from array_signal_lab import (
    ULA,
    complex_gaussian,
    condition_number,
    diagonal_load,
    sample_covariance,
    simulate_snapshots,
    steering_vector,
    theoretical_covariance,
)

ULA16 = ULA(16)
ANGLES = np.deg2rad([0.0, 30.0, -40.0])
POWERS = [1.0, 100.0, 10.0]


def _snapshots(k, seed=7):
    return simulate_snapshots(ULA16, ANGLES, POWERS, 1.0, k, np.random.default_rng(seed))


def test_complex_gaussian_power_and_circularity():
    z = complex_gaussian(np.random.default_rng(0), 400_000, power=3.0)
    assert np.mean(np.abs(z) ** 2) == pytest.approx(3.0, rel=0.01)
    # circular: E[z^2] = 0  (std of sample mean of z^2 is ~ 3/sqrt(4e5) = 0.005)
    assert abs(np.mean(z**2)) < 0.03


def test_covariance_dimensions():
    X = _snapshots(50)
    assert X.shape == (16, 50)
    assert sample_covariance(X).shape == (16, 16)
    assert theoretical_covariance(ULA16, ANGLES, POWERS, 1.0).shape == (16, 16)


def test_sample_covariance_hermitian_and_psd():
    R = sample_covariance(_snapshots(10))  # K < N: rank deficient but PSD
    np.testing.assert_array_equal(R, R.conj().T)
    ev = np.linalg.eigvalsh(R)
    assert ev.min() > -1e-10 * ev.max()
    assert np.sum(ev > 1e-9 * ev.max()) == 10  # rank K


def test_theoretical_covariance_structure():
    R = theoretical_covariance(ULA16, ANGLES, POWERS, 2.0)
    np.testing.assert_allclose(R, R.conj().T, atol=0)
    np.testing.assert_allclose(np.diag(R).real, sum(POWERS) + 2.0)
    ev = np.linalg.eigvalsh(R)
    # three signal eigenvalues above noise floor, N-3 at exactly sigma^2
    np.testing.assert_allclose(ev[:-3], 2.0, atol=1e-10)


def test_sample_covariance_converges_to_theory():
    Rt = theoretical_covariance(ULA16, ANGLES, POWERS, 1.0)
    errs = []
    for k in (100, 10_000):
        R = sample_covariance(_snapshots(k, seed=3))
        errs.append(np.linalg.norm(R - Rt) / np.linalg.norm(Rt))
    # Frobenius error scales as K^{-1/2}: factor 10 expected for 100x snapshots
    assert errs[1] < errs[0] / 5
    assert errs[1] < 0.03


def test_deterministic_with_seed():
    np.testing.assert_array_equal(_snapshots(20, seed=11), _snapshots(20, seed=11))
    assert not np.array_equal(_snapshots(20, seed=11), _snapshots(20, seed=12))


def test_diagonal_loading_and_condition_number():
    R = sample_covariance(_snapshots(8))
    assert condition_number(R) > 1e12 or np.isinf(condition_number(R))
    Rl = diagonal_load(R, 1.0)
    np.testing.assert_allclose(np.linalg.eigvalsh(Rl), np.linalg.eigvalsh(R) + 1.0, atol=1e-9)
    assert condition_number(Rl) < 1e4


def test_noise_only_snapshots():
    X = simulate_snapshots(ULA16, [], [], 2.0, 50_000, np.random.default_rng(5))
    R = sample_covariance(X)
    np.testing.assert_allclose(np.diag(R).real, 2.0, rtol=0.05)


def test_single_source_rank_one_part():
    th = np.deg2rad(10.0)
    a = steering_vector(ULA16, th)
    R = theoretical_covariance(ULA16, [th], [5.0], 0.0)
    np.testing.assert_allclose(R, 5.0 * np.outer(a, a.conj()), atol=1e-12)

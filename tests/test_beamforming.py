import numpy as np
import pytest

from array_signal_lab import (
    ULA,
    SingularCovarianceError,
    conventional_weights,
    mvdr_weights,
    response,
    sample_covariance,
    simulate_snapshots,
    steering_vector,
    theoretical_covariance,
)
from array_signal_lab.metrics import (
    conventional_sinr_single_jammer,
    optimal_sinr,
    optimal_sinr_single_jammer,
    output_sinr,
    relative_response_db,
    rmb_mean_normalized_sinr,
    white_noise_gain,
)

ULA16 = ULA(16)
TH_S = np.deg2rad(0.0)
TH_J = np.deg2rad(35.0)  # 30 deg is an exact conventional null for N = 16, d = lambda/2


def test_conventional_normalization():
    a = steering_vector(ULA16, TH_S)
    w = conventional_weights(ULA16, TH_S)
    assert np.vdot(w, a) == pytest.approx(1.0)
    assert white_noise_gain(w, a) == pytest.approx(16.0)


def test_mvdr_identity_covariance_equals_conventional():
    a = steering_vector(ULA16, np.deg2rad(17.0))
    w = mvdr_weights(np.eye(16) * 3.7, a)
    np.testing.assert_allclose(w, a / 16, atol=1e-14)


@pytest.mark.parametrize("loading", [0.0, 1.0, 100.0])
def test_mvdr_distortionless(loading):
    rng = np.random.default_rng(2)
    X = simulate_snapshots(ULA16, [TH_S, TH_J], [1.0, 1e4], 1.0, 64, rng)
    a0 = steering_vector(ULA16, TH_S)
    w = mvdr_weights(sample_covariance(X), a0, loading=loading)
    assert np.vdot(w, a0) == pytest.approx(1.0 + 0j, abs=1e-10)


def test_mvdr_clairvoyant_reaches_optimal_sinr_single_jammer():
    snr, jnr = 1.0, 1e4
    a_s = steering_vector(ULA16, TH_S)
    a_j = steering_vector(ULA16, TH_J)
    R_in = theoretical_covariance(ULA16, [TH_J], [jnr], 1.0)
    w = mvdr_weights(R_in, a_s)
    sinr = output_sinr(w, a_s, snr, R_in)
    beta = np.vdot(a_s, a_j)
    assert sinr == pytest.approx(optimal_sinr_single_jammer(16, snr, jnr, beta), rel=1e-10)
    assert sinr == pytest.approx(optimal_sinr(a_s, snr, R_in), rel=1e-10)
    # conventional SINR closed form
    wc = conventional_weights(ULA16, TH_S)
    assert output_sinr(wc, a_s, snr, R_in) == pytest.approx(
        conventional_sinr_single_jammer(16, snr, jnr, beta), rel=1e-12)


def test_mvdr_jammer_null_depth_matches_closed_form():
    # single jammer: w^H a_j = (a_s^H a_j)^* / [(N - jnr|b|^2/(1+N jnr)) (1 + N jnr)] (sigma^2 = 1)
    jnr = 1e4
    a_s = steering_vector(ULA16, TH_S)
    a_j = steering_vector(ULA16, TH_J)
    R = theoretical_covariance(ULA16, [TH_J], [jnr], 1.0)
    w = mvdr_weights(R, a_s)
    b = np.vdot(a_s, a_j)
    expected = np.abs(b) / ((16 - jnr * abs(b) ** 2 / (1 + 16 * jnr)) * (1 + 16 * jnr))
    assert np.abs(np.vdot(w, a_j)) == pytest.approx(expected, rel=1e-8)


def test_jammer_suppression_controlled_case():
    rng = np.random.default_rng(10)
    jnr = 1e4  # 40 dB
    X = simulate_snapshots(ULA16, [TH_J], [jnr], 1.0, 320, rng)  # signal-free training, K = 20N
    a_s = steering_vector(ULA16, TH_S)
    a_j = steering_vector(ULA16, TH_J)
    w = mvdr_weights(sample_covariance(X), a_s)
    wc = conventional_weights(ULA16, TH_S)
    # conventional jammer response is a sidelobe, MVDR should be far below it
    assert relative_response_db(w, a_s, a_j) < relative_response_db(wc, a_s, a_j) - 30
    R_in = theoretical_covariance(ULA16, [TH_J], [jnr], 1.0)
    rho = output_sinr(w, a_s, 1.0, R_in) / optimal_sinr(a_s, 1.0, R_in)
    # RMB mean is (K-N+2)/(K+1) = 0.950; spread is a few percent
    assert 0.85 < rho <= 1.0


def test_mvdr_raises_on_undersampled_covariance():
    X = simulate_snapshots(ULA16, [TH_J], [100.0], 1.0, 8, np.random.default_rng(1))
    with pytest.raises(SingularCovarianceError):
        mvdr_weights(sample_covariance(X), steering_vector(ULA16, TH_S))
    # diagonal loading makes it solvable
    w = mvdr_weights(sample_covariance(X), steering_vector(ULA16, TH_S), loading=1.0)
    assert np.all(np.isfinite(w))


def test_rmb_mean():
    assert rmb_mean_normalized_sinr(32, 16) == pytest.approx(18 / 33)
    assert np.isnan(rmb_mean_normalized_sinr(10, 16))


def test_mvdr_deterministic_with_seed():
    def go():
        X = simulate_snapshots(ULA16, [TH_J], [100.0], 1.0, 40, np.random.default_rng(99))
        return mvdr_weights(sample_covariance(X), steering_vector(ULA16, TH_S))
    np.testing.assert_array_equal(go(), go())


def test_response_vectorized_matches_scalar():
    w = conventional_weights(ULA16, 0.3)
    th = np.array([-0.2, 0.1, 0.3])
    np.testing.assert_allclose(response(w, ULA16, th), [response(w, ULA16, t) for t in th])


def test_signal_present_sinr_identity():
    # For any weight vector: SINR_out = x / (1 - x), x = p_s |w^H a|^2 / (w^H R w), R = R_in + p_s a a^H
    rng = np.random.default_rng(4)
    a = steering_vector(ULA16, TH_S)
    R_in = theoretical_covariance(ULA16, [TH_J], [1e3], 1.0)
    ps = 2.0
    R = R_in + ps * np.outer(a, a.conj())
    w = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    x = ps * abs(np.vdot(w, a)) ** 2 / np.vdot(w, R @ w).real
    assert output_sinr(w, a, ps, R_in) == pytest.approx(x / (1 - x), rel=1e-12)


def test_signal_present_prediction_limits():
    from array_signal_lab.metrics import smi_signal_present_mean_normalized_sinr
    # S -> 0: reduces to the RMB mean
    assert smi_signal_present_mean_normalized_sinr(64, 16, 1e-9) == pytest.approx(
        rmb_mean_normalized_sinr(64, 16), rel=1e-6)
    # larger S -> larger loss
    assert smi_signal_present_mean_normalized_sinr(64, 16, 100.0) < smi_signal_present_mean_normalized_sinr(64, 16, 1.0)


@pytest.mark.parametrize("delta_deg,snr", [(0.5, 1.0), (1.0, 10.0), (2.5, 0.1)])
def test_mismatch_closed_forms(delta_deg, snr):
    from array_signal_lab.metrics import mismatched_mpdr_sinr, mismatched_mvdr_sinr
    a0 = steering_vector(ULA16, TH_S)
    th_true = TH_S + np.deg2rad(delta_deg)
    a_s = steering_vector(ULA16, th_true)
    R_in = theoretical_covariance(ULA16, [TH_J], [1e3], 1.0)
    R = R_in + snr * np.outer(a_s, a_s.conj())
    w_mvdr = mvdr_weights(R_in, a0)
    w_mpdr = mvdr_weights(R, a0)
    assert output_sinr(w_mvdr, a_s, snr, R_in) == pytest.approx(mismatched_mvdr_sinr(a0, a_s, snr, R_in), rel=1e-9)
    assert output_sinr(w_mpdr, a_s, snr, R_in) == pytest.approx(mismatched_mpdr_sinr(a0, a_s, snr, R_in), rel=1e-9)
    # matched case reduces to the optimum
    assert mismatched_mpdr_sinr(a_s, a_s, snr, R_in) == pytest.approx(optimal_sinr(a_s, snr, R_in), rel=1e-9)

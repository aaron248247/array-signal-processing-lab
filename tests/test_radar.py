import numpy as np
import pytest

from array_signal_lab.doppler import (
    coherent_gain,
    doppler_axis,
    doppler_process,
    scalloping_loss_db,
    window,
)
from array_signal_lab.matched_filter import (
    estimate_delay_samples,
    matched_filter,
    matched_filter_direct,
    parabolic_peak,
)
from array_signal_lab.radar_signal import PointTarget, RadarParams, simulate_cpi
from array_signal_lab.waveforms import (
    lfm,
    lfm_autocorrelation_magnitude,
    lfm_coupling_delay,
    lfm_pulse,
    rect_pulse,
)

P = RadarParams()


def test_params_derived_quantities():
    assert P.wavelength == pytest.approx(0.0999308, rel=1e-5)
    assert P.n_fast == 1000
    assert P.n_pulse_samples == 100
    assert P.range_bin == pytest.approx(14.98962, rel=1e-6)
    assert P.range_resolution == pytest.approx(29.97925, rel=1e-6)
    assert P.doppler_bin == pytest.approx(156.25)
    assert P.unambiguous_velocity == pytest.approx(P.wavelength * 2500)


def test_lfm_unit_modulus_and_frequency_sweep():
    s = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    assert s.size == 100
    np.testing.assert_allclose(np.abs(s), 1.0)
    f_inst = np.diff(np.unwrap(np.angle(s))) * P.fs_hz / (2 * np.pi)
    assert f_inst[0] == pytest.approx(-P.bandwidth_hz / 2, rel=0.03)
    assert f_inst[-1] == pytest.approx(P.bandwidth_hz / 2, rel=0.03)
    assert np.all(np.diff(f_inst) > 0)


def test_lfm_zero_outside_support():
    assert np.all(lfm(np.array([-1e-9, P.pulse_s, 2 * P.pulse_s]), P.bandwidth_hz, P.pulse_s) == 0)


def test_fft_matched_filter_equals_direct():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300) + 1j * rng.standard_normal(300)
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    np.testing.assert_allclose(matched_filter(x, h), matched_filter_direct(x, h), atol=1e-11)


@pytest.mark.parametrize("delay", [0, 1, 137, 899])
def test_integer_delay_peak_at_expected_sample(delay):
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    x = np.zeros(1000, dtype=complex)
    x[delay:delay + h.size] = 3.0 * h
    y = matched_filter(x, h)
    assert int(np.argmax(np.abs(y))) == delay
    assert np.abs(y[delay]) == pytest.approx(3.0 * np.sqrt(h.size), rel=1e-12)
    assert estimate_delay_samples(y) == pytest.approx(delay, abs=1e-9)


def test_matched_filter_preserves_white_noise_variance():
    rng = np.random.default_rng(1)
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    x = (rng.standard_normal((200, 1000)) + 1j * rng.standard_normal((200, 1000))) / np.sqrt(2)
    y = matched_filter(x, h, axis=1)[:, : 1000 - h.size]  # full-overlap lags only
    assert np.mean(np.abs(y) ** 2) == pytest.approx(1.0, rel=0.01)


def test_rect_pulse_matched_filter_triangle():
    h = rect_pulse(10e-6, 1e6)  # 10 samples
    x = np.zeros(50, dtype=complex)
    x[20:30] = 1
    y = np.abs(matched_filter(x, h)) * np.linalg.norm(h)
    np.testing.assert_allclose(y[15:26], [5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5], atol=1e-12)


def test_sampled_lfm_autocorrelation_matches_closed_form():
    fs = 50e6  # heavy oversampling -> close to the continuous-time result
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, fs)
    L = h.size
    x = np.concatenate([np.zeros(L), h, np.zeros(L)])
    y = np.abs(matched_filter(x, h)) / np.sqrt(L)
    lags = (np.arange(x.size) - L) / fs
    ref = lfm_autocorrelation_magnitude(lags, P.bandwidth_hz, P.pulse_s)
    assert np.max(np.abs(y - ref)) < 0.02


def test_parabolic_peak_exact_for_parabola():
    v = -((np.arange(7) - 3.3) ** 2) + 10
    assert parabolic_peak(v, 3) == pytest.approx(0.3, abs=1e-12)
    assert parabolic_peak(v, 0) == 0.0


def test_noiseless_range_recovery_within_resolution():
    for r in (2000.0, 4321.7, 8888.8):
        X = simulate_cpi(P, [PointTarget(r, 0.0, 20.0)], include_range_migration=False)
        y = matched_filter(X[0], lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz))
        r_hat = estimate_delay_samples(y) * P.range_bin
        assert abs(r_hat - r) < P.range_resolution / 2
        r_grid = estimate_delay_samples(y, interpolate=False) * P.range_bin
        assert abs(r_grid - r) <= P.range_bin / 2 + 1e-9


def test_noiseless_doppler_recovery_within_bin():
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    fd_axis = doppler_axis(P.prf_hz, P.n_pulses)
    for v in (-200.0, -31.3, 0.0, 47.0, 120.0):
        X = simulate_cpi(P, [PointTarget(5000.0, v, 10.0)])
        RD = doppler_process(matched_filter(X, h, axis=1))
        k, l = np.unravel_index(np.argmax(np.abs(RD)), RD.shape)
        assert abs(fd_axis[k] - P.doppler_hz(v)) <= P.doppler_bin / 2 + 1e-9
        assert abs(l * P.range_bin - 5000.0) < P.range_resolution


def test_doppler_sign_convention_closing_positive():
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    for v, sign in ((100.0, 1), (-100.0, -1)):
        X = simulate_cpi(P, [PointTarget(3000.0, v, 10.0)])
        RD = doppler_process(matched_filter(X, h, axis=1))
        k, _ = np.unravel_index(np.argmax(np.abs(RD)), RD.shape)
        assert np.sign(doppler_axis(P.prf_hz, P.n_pulses)[k]) == sign


def test_on_bin_doppler_exact_and_coherent_gain():
    v = 10 * P.velocity_bin  # exactly bin 10
    r_on_bin = 200 * P.range_bin  # integer-sample delay: no range straddle loss
    X = simulate_cpi(P, [PointTarget(r_on_bin, v, 0.0)], include_range_migration=False)
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    RD = doppler_process(matched_filter(X, h, axis=1))
    k, l = np.unravel_index(np.argmax(np.abs(RD)), RD.shape)
    assert doppler_axis(P.prf_hz, P.n_pulses)[k] == pytest.approx(10 * P.doppler_bin)
    # peak amplitude: sqrt(L) (pulse compression) * sqrt(M) (rect window, unit-energy taps),
    # reduced by the intra-pulse Doppler mismatch (|chi(0, f_D)| = |sinc(f_D T)|)
    fd, L = P.doppler_hz(v), 100
    dirichlet = np.abs(np.sin(np.pi * fd * L / P.fs_hz) / (L * np.sin(np.pi * fd / P.fs_hz)))
    expected = np.sqrt(L) * np.sqrt(64) * dirichlet
    assert l == 200
    assert np.abs(RD[k, l]) == pytest.approx(expected, rel=1e-9)


def test_range_doppler_coupling_shift():
    fd = 400e3  # deliberately large to make the shift several samples
    h = lfm_pulse(P.bandwidth_hz, P.pulse_s, P.fs_hz)
    n = np.arange(1000)
    x = np.zeros(1000, dtype=complex)
    x[300:400] = h
    x *= np.exp(2j * np.pi * fd * n / P.fs_hz)
    shift = estimate_delay_samples(matched_filter(x, h)) - 300
    assert shift == pytest.approx(lfm_coupling_delay(fd, P.bandwidth_hz, P.pulse_s) * P.fs_hz, abs=0.1)


def test_scalloping_loss_known_values():
    assert scalloping_loss_db("rect", 64, 0.5) == pytest.approx(-3.92, abs=0.01)
    assert scalloping_loss_db("hann", 64, 0.5) == pytest.approx(-1.42, abs=0.03)
    assert scalloping_loss_db("rect", 64, 0.0) == pytest.approx(0.0)


def test_window_normalization_and_gain():
    for w in ("rect", "hann", "hamming"):
        assert np.sum(window(w, 64) ** 2) == pytest.approx(1.0)
    assert coherent_gain("rect", 64) == pytest.approx(64.0)
    assert coherent_gain("hann", 64) < 64.0


def test_simulation_deterministic_seed():
    tg = [PointTarget(4000, 30, -10)]
    a = simulate_cpi(P, tg, rng=np.random.default_rng(3))
    b = simulate_cpi(P, tg, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)

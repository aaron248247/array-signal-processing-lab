import numpy as np
import pytest
from scipy import stats

from array_signal_lab.cfar import (
    binomial_ci,
    ca_alpha,
    ca_cfar_1d,
    ca_cfar_2d,
    ca_pfa,
    os_alpha,
    os_cfar_1d,
    os_pfa,
    pd_ca_swerling0,
    pd_ca_swerling1,
    pd_fixed_threshold_swerling0,
    pd_fixed_threshold_swerling1,
)


@pytest.mark.parametrize("nt", [4, 16, 32, 128])
@pytest.mark.parametrize("pfa", [1e-2, 1e-4, 1e-6])
def test_ca_alpha_inverts_pfa(nt, pfa):
    assert ca_pfa(ca_alpha(nt, pfa), nt) == pytest.approx(pfa, rel=1e-10)


def test_ca_alpha_large_n_limit():
    # N_t -> inf: alpha -> -ln(pfa) (known-noise threshold)
    assert ca_alpha(10**7, 1e-4) == pytest.approx(-np.log(1e-4), rel=1e-5)
    assert ca_alpha(16, 1e-4) > -np.log(1e-4)  # CFAR loss: higher multiplier for finite N_t


@pytest.mark.parametrize("nt,k", [(16, 12), (24, 18), (8, 8)])
def test_os_alpha_inverts_pfa(nt, k):
    a = os_alpha(nt, k, 1e-4)
    assert os_pfa(a, nt, k) == pytest.approx(1e-4, rel=1e-8)


def test_ca_cfar_1d_manual_example():
    z = np.ones(21)
    z[10] = 100.0
    det, thr = ca_cfar_1d(z, n_train=4, n_guard=1, pfa=1e-3, edge="valid")
    a = ca_alpha(8, 1e-3)
    assert thr[10] == pytest.approx(a * 1.0)
    assert det[10] and det.sum() == 1
    # cells whose window overlaps the spike: mean = (7 + 100)/8
    assert thr[5] == pytest.approx(a * (7 + 100) / 8)


def test_edge_handling_valid_truncate_wrap():
    z = np.arange(1.0, 31.0)
    reach = 2 + 5
    _, tv = ca_cfar_1d(z, 5, 2, 1e-3, edge="valid")
    assert np.all(np.isnan(tv[:reach])) and np.all(np.isnan(tv[-reach:]))
    assert np.all(np.isfinite(tv[reach:-reach]))
    _, tt = ca_cfar_1d(z, 5, 2, 1e-3, edge="truncate")
    assert np.all(np.isfinite(tt))
    # first cell: only lagging cells 3..7 (values 4..8) available -> N = 5
    assert tt[0] == pytest.approx(ca_alpha(5, 1e-3) * np.mean(z[3:8]))
    _, tw = ca_cfar_1d(z, 5, 2, 1e-3, edge="wrap")
    train0 = np.r_[z[-7:-2], z[3:8]]
    assert tw[0] == pytest.approx(ca_alpha(10, 1e-3) * train0.mean())
    np.testing.assert_allclose(tv[reach:-reach], tw[reach:-reach])


def test_ca_cfar_1d_brute_force():
    rng = np.random.default_rng(0)
    z = rng.exponential(size=200)
    det, thr = ca_cfar_1d(z, 6, 2, 1e-2, edge="valid")
    a = ca_alpha(12, 1e-2)
    for i in range(8, 192):
        tr = np.r_[z[i - 8:i - 2], z[i + 3:i + 9]]
        assert thr[i] == pytest.approx(a * tr.mean(), rel=1e-12)
        assert det[i] == (z[i] > a * tr.mean())


def test_ca_cfar_2d_matches_brute_force_and_1d_limit():
    rng = np.random.default_rng(1)
    z = rng.exponential(size=(16, 40))
    det, thr, full = ca_cfar_2d(z, (2, 3), (1, 1), 1e-3, wrap_axis0=True)
    assert full == (2 * 3 + 1) * (2 * 4 + 1) - 3 * 3
    a = ca_alpha(full, 1e-3)
    for k in (0, 5, 15):
        for l in (4, 20, 35):
            rows = [(k + d) % 16 for d in range(-3, 4)]
            box = z[np.ix_(rows, range(l - 4, l + 5))].copy()
            box[2:5, 3:6] = np.nan
            assert thr[k, l] == pytest.approx(a * np.nanmean(box), rel=1e-12)
    assert np.all(np.isnan(thr[:, :4])) and np.all(np.isnan(thr[:, -4:]))
    # with zero Doppler extent the 2-D detector reduces to the 1-D one
    _, thr2, _ = ca_cfar_2d(z[:1], (0, 5), (0, 2), 1e-3, wrap_axis0=False)
    _, thr1 = ca_cfar_1d(z[:1], 5, 2, 1e-3)
    np.testing.assert_allclose(thr2, thr1)


def test_os_cfar_brute_force():
    rng = np.random.default_rng(2)
    z = rng.exponential(size=100)
    det, thr = os_cfar_1d(z, 8, 2, 12, 1e-3)
    a = os_alpha(16, 12, 1e-3)
    i = 50
    tr = np.r_[z[i - 10:i - 2], z[i + 3:i + 11]]
    assert thr[i] == pytest.approx(a * np.sort(tr)[11])
    assert np.isnan(thr[5])


def test_cfar_property_scale_invariance():
    rng = np.random.default_rng(3)
    z = rng.exponential(size=5000)
    d1, _ = ca_cfar_1d(z, 8, 2, 1e-2)
    d2, _ = ca_cfar_1d(1e6 * z, 8, 2, 1e-2)
    np.testing.assert_array_equal(d1, d2)


def test_deterministic_seed():
    a = ca_cfar_1d(np.random.default_rng(9).exponential(size=1000), 8, 2, 1e-2)[0]
    b = ca_cfar_1d(np.random.default_rng(9).exponential(size=1000), 8, 2, 1e-2)[0]
    np.testing.assert_array_equal(a, b)


def test_monte_carlo_pfa_independent_windows():
    """Independent windows -> detection count is Binomial(n, pfa); check with an exact 99.9 % CI."""
    rng = np.random.default_rng(12345)
    pfa, nt, n = 1e-3, 16, 400_000
    # each row is an independent window: 8 train, 2 guard, CUT, 2 guard, 8 train
    z = rng.exponential(size=(n, 21))
    det, _ = ca_cfar_1d(z, 8, 2, pfa, edge="valid")
    k = int(det[:, 10].sum())
    lo, hi = binomial_ci(k, n, 0.999)
    assert lo <= pfa <= hi, (k, lo, hi)


def test_monte_carlo_pfa_sliding_window():
    """Overlapping windows: marginal pfa still exact; use independent sequences for the CI."""
    rng = np.random.default_rng(54321)
    pfa = 1e-3
    z = rng.exponential(size=(400, 2000))
    det, thr = ca_cfar_1d(z, 8, 2, pfa)
    per_seq = det.sum(axis=1) / np.isfinite(thr).sum(axis=1)
    m, se = per_seq.mean(), per_seq.std(ddof=1) / np.sqrt(per_seq.size)
    assert abs(m - pfa) < 4 * se


def test_pd_closed_forms_limits():
    # SNR -> 0: Pd -> Pfa
    assert pd_ca_swerling1(1e-9, 1e-4, 16) == pytest.approx(1e-4, rel=1e-6)
    assert pd_ca_swerling0(1e-9, 1e-4, 16) == pytest.approx(1e-4, rel=1e-4)
    assert pd_fixed_threshold_swerling0(1e-9, 1e-4) == pytest.approx(1e-4, rel=1e-6)
    # N_t -> large: CA -> fixed threshold
    assert pd_ca_swerling1(20.0, 1e-4, 100000) == pytest.approx(pd_fixed_threshold_swerling1(20.0, 1e-4), rel=1e-4)
    assert pd_ca_swerling0(20.0, 1e-4, 4000) == pytest.approx(pd_fixed_threshold_swerling0(20.0, 1e-4), abs=2e-3)
    # finite N_t costs detection probability
    assert pd_ca_swerling0(20.0, 1e-4, 16) < pd_fixed_threshold_swerling0(20.0, 1e-4)


def test_swerling1_ca_closed_form_vs_monte_carlo():
    rng = np.random.default_rng(77)
    pfa, n, snr = 1e-2, 200_000, 10.0
    z = rng.exponential(size=(n, 21))
    z[:, 10] = rng.exponential(scale=1 + snr, size=n)  # |CN(0,1+snr)|^2
    det, _ = ca_cfar_1d(z, 8, 2, pfa)
    k = int(det[:, 10].sum())
    lo, hi = binomial_ci(k, n, 0.999)
    assert lo <= pd_ca_swerling1(snr, pfa, 16) <= hi


def test_binomial_ci_contains_estimate():
    lo, hi = binomial_ci(50, 10000)
    assert lo < 0.005 < hi
    assert stats.binom.cdf(50, 10000, hi) == pytest.approx(0.025, rel=1e-3)

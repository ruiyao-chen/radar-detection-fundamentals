"""Detection probability against closed forms, a direct Marcum-Q integral and Monte Carlo."""

import numpy as np
import pytest
from scipy import integrate, special

from pulseradar import detection as D

from conftest import binomial_tol


def marcum_q1(a, b):
    f = lambda x: x * np.exp(-((x - a) ** 2) / 2) * special.i0e(a * x)  # noqa: E731
    return integrate.quad(f, b, np.inf)[0]


@pytest.mark.parametrize("pfa", [1e-3, 1e-6])
@pytest.mark.parametrize("snr_db", [0.0, 6.0, 10.0, 14.0])
def test_single_pulse_is_marcum_q(pfa, snr_db):
    s = 10 ** (snr_db / 10)
    assert D.pd_single_pulse(s, pfa) == pytest.approx(marcum_q1(np.sqrt(2 * s), np.sqrt(-2 * np.log(pfa))), abs=1e-6)


@pytest.mark.parametrize("pfa", [1e-3, 1e-6])
@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0, 15.0])
def test_swerling1_single_pulse_closed_form(pfa, snr_db):
    s = 10 ** (snr_db / 10)
    assert D.pd_swerling1(s, pfa, 1) == pytest.approx(pfa ** (1 / (1 + s)), abs=1e-6)


def _mc_noncoherent(rng, snr, n, pfa, swerling, trials=60_000):
    T = D.threshold_noncoherent(pfa, n)
    s = snr * (rng.exponential(size=(trials, 1)) if swerling == 1 else 1.0)
    z = rng.standard_normal((trials, n)) + 1j * rng.standard_normal((trials, n))
    z = z / np.sqrt(2) + np.sqrt(s) * np.exp(1j * rng.uniform(0, 2 * np.pi, (trials, n)))
    return float((np.sum(np.abs(z) ** 2, axis=1) > T).mean()), trials


@pytest.mark.parametrize("swerling", [0, 1])
@pytest.mark.parametrize("snr_db", [-4.0, 0.0, 3.0])
def test_noncoherent_monte_carlo(rng, swerling, snr_db):
    n, pfa, s = 10, 1e-4, 10 ** (snr_db / 10)
    pd_mc, trials = _mc_noncoherent(rng, s, n, pfa, swerling)
    f = D.pd_swerling0 if swerling == 0 else D.pd_swerling1
    p = float(f(s, pfa, n))
    assert abs(pd_mc - p) < binomial_tol(trials, p) + 1e-4


def test_noise_alone_gives_design_pfa(rng):
    n, pfa, trials = 8, 1e-3, 400_000
    T = D.threshold_noncoherent(pfa, n)
    z = (rng.standard_normal((trials, n)) + 1j * rng.standard_normal((trials, n))) / np.sqrt(2)
    assert abs((np.sum(np.abs(z) ** 2, axis=1) > T).mean() - pfa) < binomial_tol(trials, pfa)


def test_albersheim_round_trip():
    for n in (1, 10, 150):
        snr = D.albersheim_snr_db(0.8, 1e-6, n)
        assert D.albersheim_pd(snr, 1e-6, n) == pytest.approx(0.8, abs=1e-9)


@pytest.mark.parametrize("pfa", [1e-4, 1e-6, 1e-7])
@pytest.mark.parametrize("n", [1, 10, 100])
@pytest.mark.parametrize("pd", [0.5, 0.7, 0.9])
def test_albersheim_close_to_exact_in_the_design_region(pfa, n, pd):
    exact = 10 * np.log10(D.snr_required(pd, pfa, n, 0))
    assert abs(D.albersheim_snr_db(pd, pfa, n) - exact) < 0.35


def test_albersheim_overstates_pd_at_low_snr():
    # Solved for Pd, the approximation says 0.10 at 0 dB, 1e-3; the exact answer is 0.0185
    assert D.albersheim_pd(0.0, 1e-3, 1) == pytest.approx(0.10, abs=0.005)
    assert D.pd_single_pulse(1.0, 1e-3) == pytest.approx(0.0185, abs=0.0005)


def test_snr_required_inverts_pd():
    for sw in (0, 1):
        s = D.snr_required(0.8, 1e-6, 20, sw)
        f = D.pd_swerling0 if sw == 0 else D.pd_swerling1
        assert float(f(s, 1e-6, 20)) == pytest.approx(0.8, abs=1e-5)


def test_fluctuation_loss_is_positive():
    assert D.snr_required(0.9, 1e-6, 10, 1) > D.snr_required(0.9, 1e-6, 10, 0)


def test_m_of_n():
    p = 0.3
    brute = sum(special.comb(5, k) * p**k * (1 - p) ** (5 - k) for k in range(2, 6))
    assert D.m_of_n(p, 2, 5) == pytest.approx(brute)
    assert D.m_of_n(p, 1, 1) == pytest.approx(p)


def test_pfa_from_false_alarm_time():
    assert D.pfa_from_false_alarm_time(10e6, 100.0) == pytest.approx(1e-9)

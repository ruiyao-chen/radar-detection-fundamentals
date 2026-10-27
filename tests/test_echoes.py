"""Part 2: noise estimation, threshold, M-of-N, SNR/RCS estimates and CA-CFAR."""

import numpy as np
import pytest

from pulseradar import detection as D, echoes as E, radar_eq as RE
from pulseradar.config import BASELINE

from conftest import binomial_tol


def test_noise_estimators_unbiased_and_ml_tighter(rng):
    est_mean, est_ml = [], []
    for _ in range(400):
        r = E.simulate(50, 20, sigma=2.0, rng=rng)
        est_mean.append(E.sigma_from_mean(r))
        est_ml.append(E.sigma_ml(r))
    assert np.mean(est_mean) == pytest.approx(2.0, rel=0.005)
    assert np.mean(est_ml) == pytest.approx(2.0, rel=0.005)
    assert np.std(est_ml) < np.std(est_mean)


@pytest.mark.parametrize("pfa", [1e-2, 1e-3, 1e-4])
def test_threshold_gives_design_pfa(rng, pfa):
    r = E.simulate(1000, 1000, sigma=1.0, rng=rng)
    measured = E.crossings(r, E.threshold(1.0, pfa)).mean()
    assert abs(measured - pfa) < binomial_tol(r.size, pfa)


@pytest.mark.parametrize("snr_db", [3.0, 8.0, 12.0])
def test_single_pulse_pd_matches_marcum(rng, snr_db):
    s, pfa, n = 10 ** (snr_db / 10), 1e-3, 200_000
    r = E.simulate(1, n, [E.Echo(0, s)], rng=rng)
    p = float(D.pd_single_pulse(s, pfa))
    assert abs(E.crossings(r, E.threshold(1.0, pfa)).mean() - p) < binomial_tol(n, p)


def test_m_of_n_matches_binomial(rng):
    s, pfa, m, n, gates = 10 ** 0.5, 1e-2, 3, 8, 40_000
    r = E.simulate(gates, n, [E.Echo(g, s) for g in range(gates)], rng=rng)
    hits = E.crossings(r, E.threshold(1.0, pfa))
    p = float(D.m_of_n(D.pd_single_pulse(s, pfa), m, n))
    assert abs(E.m_of_n_detect(hits, m).mean() - p) < binomial_tol(gates, p)


def test_swerling1_noncoherent_matches_exact(rng):
    s, pfa, n, dwells = 10 ** 0.3, 1e-4, 10, 20_000
    T = D.threshold_noncoherent(pfa, n)
    det = 0
    for _ in range(dwells // 1000):
        r = E.simulate(1000, n, [E.Echo(g, s, swerling=1) for g in range(1000)], sigma=1 / np.sqrt(2), rng=rng)
        det += int((np.sum(r**2, axis=1) > T).sum())
    p = float(D.pd_swerling1(s, pfa, n))
    assert abs(det / dwells - p) < binomial_tol(dwells, p)


def test_snr_and_rcs_recovered(rng):
    R, rcs = 30e3, 3.0
    s = float(RE.snr_single_pulse(BASELINE, rcs, R))
    r = E.simulate(1, 4000, [E.Echo(0, s)], sigma=1.0, rng=rng)
    s_hat = E.estimate_snr(r[0], 1.0)
    assert s_hat == pytest.approx(s, rel=0.03)
    assert RE.rcs_from_snr(BASELINE, s_hat, R) == pytest.approx(rcs, rel=0.03)


def test_ca_cfar_holds_pfa_across_a_noise_step(rng):
    pfa = 1e-3
    sigma = np.where(np.arange(400) < 200, 1.0, np.sqrt(10.0))       # +10 dB from gate 200
    r = E.simulate(400, 500, sigma=sigma, rng=rng)
    det, _ = E.ca_cfar(r, pfa, guard=2, train=16)
    for sl in (slice(20, 180), slice(220, 380)):                       # away from the step
        n = det[sl].size
        assert abs(det[sl].mean() - pfa) < binomial_tol(n, pfa, k=5)
    fixed = E.crossings(r, E.threshold(1.0, pfa))
    assert fixed[220:380].mean() > 100 * pfa                          # a fixed threshold floods


@pytest.mark.parametrize("K", [10, 100])
def test_estimated_noise_inflates_pfa_as_predicted(rng, K):
    """With sigma^2 estimated from K samples, the average Pfa is (1 + ln(1/Pfa)/K)^-K, not Pfa."""
    pfa, trials, n_test = 1e-2, 20_000, 200
    q = E.simulate(trials, K, sigma=1.0, rng=rng)
    sig = np.sqrt(np.mean(q**2, axis=1) / 2.0)                       # sigma_ml per trial
    t = E.simulate(trials, n_test, sigma=1.0, rng=rng)
    measured = float((t > E.threshold(sig, pfa)[:, None]).mean())
    theory = (1.0 + np.log(1.0 / pfa) / K) ** (-K)
    assert measured == pytest.approx(theory, rel=0.04)
    assert theory > pfa

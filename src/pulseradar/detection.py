"""Detection probability: the closed-form approximation and the exact answers.

* ``pfa_from_false_alarm_time``: P_fa = 1 / (B T_fa), one decision per
  resolution cell per 1/B seconds (Skolnik).
* Albersheim's equation: the SNR per pulse needed for (Pd, Pfa) with n pulses
  noncoherently integrated, steady (Swerling 0) target. An empirical fit,
  quoted as accurate to about 0.2 dB for 1e-7 <= Pfa <= 1e-3,
  0.1 <= Pd <= 0.9 and 1 <= n <= 8096.
* Exact Swerling 0, square-law detector: with unit noise power per pulse the
  sum of n detected samples, doubled, is chi-square with 2n degrees of freedom
  under noise alone and noncentral chi-square with noncentrality 2 n SNR with
  a target.
* Exact Swerling 1 (scan-to-scan fluctuation): the RCS is exponential and
  constant over one dwell, so Pd is the Swerling 0 Pd averaged over
  SNR x Exp(1).
* M-of-N: a target is declared if at least M of N independent single-pulse
  decisions cross the threshold; binomial in the single-pulse probability.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, stats


def pfa_from_false_alarm_time(bandwidth_hz: float, false_alarm_time_s: float) -> float:
    return 1.0 / (bandwidth_hz * false_alarm_time_s)


# --------------------------------------------------------------------------- #
# Albersheim
# --------------------------------------------------------------------------- #
def _alb_parts(pfa, n):
    A = np.log(0.62 / pfa)
    k = 6.2 + 4.54 / np.sqrt(n + 0.44)
    return A, k


def albersheim_snr_db(pd, pfa, n=1):
    """Required SNR per pulse (dB), steady target, n pulses noncoherently integrated."""
    A, k = _alb_parts(pfa, n)
    B = np.log(pd / (1.0 - pd))
    return -5.0 * np.log10(n) + k * np.log10(A + 0.12 * A * B + 1.7 * B)


def albersheim_pd(snr_db, pfa, n=1):
    """Albersheim's equation solved for Pd (the form used in many design spreadsheets)."""
    A, k = _alb_parts(pfa, n)
    z = (np.asarray(snr_db, dtype=float) + 5.0 * np.log10(n)) / k
    B = (10.0**z - A) / (1.7 + 0.12 * A)
    return 1.0 / (1.0 + np.exp(-B))


# --------------------------------------------------------------------------- #
# Exact
# --------------------------------------------------------------------------- #
def threshold_noncoherent(pfa, n=1):
    """Threshold on the sum of n square-law samples (unit noise power per pulse)."""
    return 0.5 * stats.chi2.isf(pfa, 2 * n)


def pd_swerling0(snr, pfa, n=1):
    """Exact Pd, steady target, square-law detector, n pulses noncoherently integrated."""
    T = threshold_noncoherent(pfa, n)
    snr = np.asarray(snr, dtype=float)
    return stats.ncx2.sf(2.0 * T, 2 * n, 2.0 * n * snr)


def pd_swerling1(snr, pfa, n=1):
    """Exact Pd, Swerling 1: Swerling 0 averaged over an exponential RCS."""
    T = threshold_noncoherent(pfa, n)

    def one(s):
        if s <= 0:
            return float(pfa)
        f = lambda x: stats.ncx2.sf(2.0 * T, 2 * n, 2.0 * n * s * x) * np.exp(-x)  # noqa: E731
        val, _ = integrate.quad(f, 0.0, np.inf, limit=200)
        return val

    snr = np.asarray(snr, dtype=float)
    return np.vectorize(one, otypes=[float])(snr)


def pd_single_pulse(snr, pfa):
    """Single pulse, steady target: Marcum Q_1(sqrt(2 SNR), sqrt(-2 ln Pfa))."""
    return pd_swerling0(snr, pfa, 1)


def snr_required(pd, pfa, n=1, swerling: int = 0) -> float:
    """SNR per pulse (linear) that gives ``pd``, by root-finding on the exact Pd."""
    f = pd_swerling0 if swerling == 0 else pd_swerling1
    g = lambda s_db: float(f(10.0 ** (s_db / 10.0), pfa, n)) - pd  # noqa: E731
    return 10.0 ** (optimize.brentq(g, -40.0, 80.0, xtol=1e-6) / 10.0)


def m_of_n(p, m: int, n: int):
    """P(at least m of n independent trials succeed)."""
    return stats.binom.sf(m - 1, n, np.asarray(p, dtype=float))

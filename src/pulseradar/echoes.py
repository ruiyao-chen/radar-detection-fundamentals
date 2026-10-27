"""Part 2: threshold detection in a range-gate x pulse matrix.

The receiver output after a linear envelope detector is sampled once per range
gate per pulse, filling a matrix whose rows are range gates (fast time) and
columns are pulses (slow time). With noise alone each sample is Rayleigh with
scale sigma (sigma^2 = noise power per I/Q channel); with a steady echo of
amplitude a it is Rician, and the single-pulse SNR is a^2 / (2 sigma^2).

Steps, each checked against an exact answer in tests/:

1. Estimate sigma from receiver-only data (transmitter off). Two estimators:
   the mean, sigma = mean(r) sqrt(2 / pi), and the maximum-likelihood one,
   sigma^2 = mean(r^2) / 2.
2. Set the threshold for a required Pfa: P(r > V_T) = exp(-V_T^2 / (2 sigma^2)),
   so V_T = sigma sqrt(2 ln(1 / Pfa)).
3. Compare every sample with V_T; count crossings per gate over the pulses and
   declare a target where at least M of N cross (binary integration).
4. Estimate the echo power in a detected gate as mean(r^2) - 2 sigma^2, hence
   its SNR, and invert the radar equation for its radar cross-section.
5. When the noise floor changes with range, a single threshold no longer holds
   Pfa. Cell-averaging CFAR estimates the local noise from neighbouring gates
   instead; for square-law samples averaged over N cells the threshold
   multiplier alpha = N (Pfa^(-1/N) - 1) gives exactly the design Pfa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Echo:
    gate: int
    snr: float            # mean single-pulse SNR (linear)
    swerling: int = 0     # 0 steady; 1 exponential RCS, constant over the dwell


def simulate(n_gates: int, n_pulses: int, echoes=(), sigma=1.0, rng=None) -> np.ndarray:
    """Envelope-detected samples, shape (n_gates, n_pulses).

    ``sigma`` is a scalar or a per-gate array (noise floor varying with range).
    """
    rng = np.random.default_rng() if rng is None else rng
    sig = np.broadcast_to(np.asarray(sigma, dtype=float).reshape(-1, 1), (n_gates, 1))
    z = sig * (rng.standard_normal((n_gates, n_pulses)) + 1j * rng.standard_normal((n_gates, n_pulses)))
    for e in echoes:
        snr = e.snr * (rng.exponential() if e.swerling == 1 else 1.0)
        a = sig[e.gate, 0] * np.sqrt(2.0 * snr)
        z[e.gate] += a * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, n_pulses))
    return np.abs(z)


def sigma_from_mean(r) -> float:
    return float(np.mean(r) * np.sqrt(2.0 / np.pi))


def sigma_ml(r) -> float:
    return float(np.sqrt(np.mean(np.square(r)) / 2.0))


def threshold(sigma, pfa: float):
    return np.asarray(sigma, dtype=float) * np.sqrt(2.0 * np.log(1.0 / pfa))


def crossings(r, v_t) -> np.ndarray:
    return np.asarray(r) > v_t


def m_of_n_detect(hits: np.ndarray, m: int) -> np.ndarray:
    """Per gate: at least m crossings over the pulses (axis 1)."""
    return hits.sum(axis=1) >= m


def estimate_snr(r_gate, sigma: float) -> float:
    """Mean single-pulse SNR of one gate from its samples: (mean r^2 - 2 sigma^2) / (2 sigma^2)."""
    return float((np.mean(np.square(r_gate)) - 2.0 * sigma**2) / (2.0 * sigma**2))


def ca_cfar(r, pfa: float, guard: int = 2, train: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Cell-averaging CFAR along range (axis 0) on square-law samples x = r^2.

    For each gate the noise is the mean of up to ``train`` gates on each side,
    beyond ``guard`` gates; the multiplier uses the number of training cells
    actually available, so edges keep the design Pfa. Returns (detections,
    threshold on r).
    """
    x = np.square(np.asarray(r, dtype=float))
    G = x.shape[0]
    det = np.zeros_like(x, dtype=bool)
    thr = np.zeros_like(x)
    for g in range(G):
        lo = list(range(max(0, g - guard - train), max(0, g - guard)))
        hi = list(range(min(G, g + guard + 1), min(G, g + guard + 1 + train)))
        idx = lo + hi
        n = len(idx)
        alpha = n * (pfa ** (-1.0 / n) - 1.0)
        t = alpha * x[idx].mean(axis=0)
        thr[g] = np.sqrt(t)
        det[g] = x[g] > t
    return det, thr

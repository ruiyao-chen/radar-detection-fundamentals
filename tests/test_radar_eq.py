"""Radar equation: scaling laws, the thermal-noise anchor, and the RCS inverse."""

import numpy as np
import pytest

from pulseradar import radar_eq as RE
from pulseradar.config import BASELINE


def test_noise_anchor_kTB():
    # k T0 B with F = 0 dB and B = 1 MHz is -114 dBm (the standard textbook figure)
    d = BASELINE.with_(noise_figure_db=0.0, bandwidth_hz=1e6)
    assert RE.db(RE.noise_power_w(d)) + 30 == pytest.approx(-113.98, abs=0.01)


def test_scaling_laws():
    s1 = RE.snr_single_pulse(BASELINE, 1.0, 20e3)
    assert RE.snr_single_pulse(BASELINE, 1.0, 40e3) == pytest.approx(s1 / 16)
    assert RE.snr_single_pulse(BASELINE, 4.0, 20e3) == pytest.approx(4 * s1)
    assert RE.snr_single_pulse(BASELINE.with_(peak_power_w=2 * BASELINE.peak_power_w), 1.0, 20e3) == pytest.approx(2 * s1)
    # energy, not bandwidth: changing the chirp bandwidth leaves SNR unchanged
    assert RE.snr_single_pulse(BASELINE.with_(bandwidth_hz=50e6), 1.0, 20e3) == pytest.approx(s1)


def test_atmospheric_loss_is_two_way_db_per_km():
    s0 = RE.snr_single_pulse(BASELINE, 1.0, 50e3, 0.0)
    s1 = RE.snr_single_pulse(BASELINE, 1.0, 50e3, 0.02)
    assert RE.db(s0 / s1) == pytest.approx(1.0, abs=1e-9)


def test_rcs_inverse():
    R = np.array([10e3, 30e3, 60e3])
    snr = RE.snr_single_pulse(BASELINE, 7.5, R, 0.02)
    assert RE.rcs_from_snr(BASELINE, snr, R, 0.02) == pytest.approx(7.5)


def test_gain_rule_of_thumb():
    assert RE.db(RE.antenna_gain(1.0, 1.0)) == pytest.approx(44.15, abs=0.01)

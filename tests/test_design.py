"""Clutter, Doppler and the design evaluator."""

import numpy as np
import pytest

from pulseradar import clutter, design, doppler
from pulseradar.config import BASELINE, SCENARIO, C


def test_constant_gamma_at_normal_incidence():
    assert clutter.sigma0_constant_gamma(np.pi / 2, -15.0) == pytest.approx(10 ** -1.5)


def test_clutter_cell_pulse_vs_beam_limited():
    az, el, dr = np.radians(1.0), np.radians(5.0), 10.0
    low = clutter.clutter_area_m2(30e3, np.radians(3.0), az, el, dr)
    assert low == pytest.approx(30e3 * az * dr / np.cos(np.radians(3.0)))
    steep = clutter.clutter_area_m2(1e3, np.radians(85.0), az, el, 1e4)
    assert steep == pytest.approx(1e3 * az * 1e3 * el / np.sin(np.radians(85.0)))


def test_doppler_relations():
    lam, prf = 0.03, 5_000.0
    assert doppler.blind_speed_ms(prf, lam) == pytest.approx(75.0)
    assert doppler.unambiguous_range_m(1_000.0) == pytest.approx(C / 2_000.0)
    v, az, psi, beam = 100.0, np.pi / 2, np.radians(10.0), np.radians(1.2)
    spread = doppler.clutter_spread_hz(v, beam, az, psi, lam)
    assert doppler.mdv_ms(v, beam, az, psi) == pytest.approx(lam / 2 * spread / 2)
    assert doppler.clutter_doppler_hz(v, az, psi, lam) == pytest.approx(0.0, abs=1e-9)   # broadside


def test_baseline_meets_requirements():
    m = design.evaluate(BASELINE, SCENARIO)
    assert m["feasible"]
    assert m["range_ambiguities"] >= 1        # medium PRF: the far edge is range-ambiguous (reported, not required)


def test_low_power_fails_pd():
    m = design.evaluate(BASELINE.with_(peak_power_w=50.0), SCENARIO)
    failed = [c["check"] for c in m["checks"] if not c["pass"] and c["required"]]
    assert failed == ["Pd at far edge, Swerling 1"]


def test_faster_scan_fewer_pulses_shorter_revisit():
    a = design.evaluate(BASELINE.with_(scan_rate_deg_s=20.0), SCENARIO)
    b = design.evaluate(BASELINE.with_(scan_rate_deg_s=80.0), SCENARIO)
    assert b["pulses_on_target"] < a["pulses_on_target"] and b["revisit_s"] < a["revisit_s"]
    assert b["pd_far"] < a["pd_far"]


def test_trade_study_flags():
    df = design.trade_study(BASELINE, SCENARIO, [2_000.0, 5_000.0, 9_000.0], [20.0, 40.0, 120.0])
    assert df.loc[df.prf_hz == 2_000.0, "fail_doppler"].all()      # below 4 v_max / lambda
    assert df.loc[df.prf_hz == 9_000.0, "fail_duty"].all()         # 20 us x 9 kHz = 18 %
    assert df.loc[df.scan_rate_deg_s == 20.0, "fail_revisit"].all()  # 120 deg / 20 deg/s = 6 s

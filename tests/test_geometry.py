"""4/3-Earth geometry: round trips, the flat-Earth limit, the horizon."""

import numpy as np
import pytest

from pulseradar import geometry as G


@pytest.mark.parametrize("h", [500.0, 6_000.0, 12_000.0])
def test_round_trips(h):
    ground = np.array([1e3, 10e3, 50e3, 150e3])
    R = G.slant_from_ground(ground, h)
    assert G.ground_from_slant(R, h) == pytest.approx(ground, rel=1e-9)
    psi = G.grazing_from_slant(R, h)
    assert G.slant_from_grazing(psi, h) == pytest.approx(R, rel=1e-9)


def test_flat_earth_limit_at_short_range():
    h, g = 1_000.0, 2_000.0
    assert G.slant_from_ground(g, h) == pytest.approx(np.hypot(h, g), rel=1e-4)
    assert np.degrees(G.grazing_from_slant(np.hypot(h, g), h)) == pytest.approx(np.degrees(np.arctan2(h, g)), abs=0.02)


def test_curvature_lowers_grazing_at_long_range():
    h, g = 6_000.0, 50_000.0
    flat = np.degrees(np.arctan2(h, g))
    assert np.degrees(G.grazing_from_slant(G.slant_from_ground(g, h), h)) < flat


def test_horizon_has_zero_grazing():
    h = 6_000.0
    assert G.grazing_from_slant(G.horizon_m(h), h) == pytest.approx(0.0, abs=1e-6)

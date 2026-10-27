"""Doppler, PRF and range ambiguity for a moving platform looking at the ground.

Ground clutter in the beam at azimuth phi (from the velocity vector) and
grazing angle psi has Doppler f_c = 2 V cos(phi) cos(psi) / lambda. Across an
azimuth beam of width theta_az it spreads over

    delta_f = (2 V / lambda) theta_az sin(phi) cos(psi),

widest at broadside. After the processor centres the clutter at zero Doppler,
a target is outside the main-lobe clutter (exo-clutter, noise-limited) when its
radial speed exceeds the minimum detectable velocity
MDV = (lambda / 2)(delta_f / 2) = V theta_az sin(phi) cos(psi) / 2.

The PRF samples the Doppler spectrum. To measure radial speeds up to +/- v_max
relative to the clutter without the clutter band folding onto them, the PRF
must exceed 4 v_max / lambda + delta_f. Blind speeds fall at multiples of
lambda PRF / 2. Ranges beyond c / (2 PRF) fold back (range ambiguity).
"""

from __future__ import annotations

import numpy as np

from .config import C


def doppler_hz(radial_speed_ms, wavelength_m: float):
    return 2.0 * np.asarray(radial_speed_ms, dtype=float) / wavelength_m


def clutter_doppler_hz(v_ms, az_rad, psi_rad, wavelength_m):
    return 2.0 * v_ms * np.cos(az_rad) * np.cos(psi_rad) / wavelength_m


def clutter_spread_hz(v_ms, beam_az_rad, az_rad, psi_rad, wavelength_m):
    return 2.0 * v_ms * beam_az_rad * np.abs(np.sin(az_rad)) * np.cos(psi_rad) / wavelength_m


def mdv_ms(v_ms, beam_az_rad, az_rad, psi_rad):
    return 0.5 * v_ms * beam_az_rad * np.abs(np.sin(az_rad)) * np.cos(psi_rad)


def min_prf_hz(v_max_ms, spread_hz, wavelength_m):
    return 4.0 * v_max_ms / wavelength_m + spread_hz


def blind_speed_ms(prf_hz, wavelength_m, k: int = 1):
    return k * wavelength_m * prf_hz / 2.0


def unambiguous_range_m(prf_hz):
    return C / (2.0 * np.asarray(prf_hz, dtype=float))


def n_range_ambiguities(max_range_m, prf_hz):
    """How many PRIs the farthest echo arrives after (0 = unambiguous)."""
    return np.floor(np.asarray(max_range_m, dtype=float) / unambiguous_range_m(prf_hz)).astype(int)

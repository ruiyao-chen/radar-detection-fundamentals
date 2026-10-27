"""Land clutter seen by a look-down radar.

Constant-gamma model: sigma0 = gamma sin(psi). gamma is a property of the
terrain (about -10 to -20 dB for farmland and wooded hills at X band in the
standard references); -15 dB here is assumed, not measured.

The clutter in one resolution cell is sigma0 times the illuminated area. At
low grazing angles the cell is pulse-limited (range resolution projected on
the ground, times the azimuth beam width); at steep angles it is
beam-limited (the whole elevation footprint). The smaller of the two applies.
"""

from __future__ import annotations

import numpy as np


def sigma0_constant_gamma(psi_rad, gamma_db: float):
    return 10.0 ** (gamma_db / 10.0) * np.sin(psi_rad)


def clutter_area_m2(range_m, psi_rad, beam_az_rad: float, beam_el_rad: float, range_res_m: float):
    R = np.asarray(range_m, dtype=float)
    psi = np.asarray(psi_rad, dtype=float)
    pulse_limited = R * beam_az_rad * range_res_m / np.cos(psi)
    beam_limited = R * beam_az_rad * R * beam_el_rad / np.sin(psi)
    return np.minimum(pulse_limited, beam_limited)


def clutter_rcs_m2(range_m, psi_rad, beam_az_rad, beam_el_rad, range_res_m, gamma_db):
    return sigma0_constant_gamma(psi_rad, gamma_db) * clutter_area_m2(range_m, psi_rad, beam_az_rad,
                                                                      beam_el_rad, range_res_m)

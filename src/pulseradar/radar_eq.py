"""The radar range equation and its inverse.

Single-pulse SNR after a matched filter depends on the transmitted energy
P_t * tau, not on the bandwidth (pulse compression trades a long pulse for a
short compressed one without changing the energy):

    SNR = P_t tau G^2 lambda^2 sigma / ((4 pi)^3 R^4 k T0 F L_sys L_atm(R))

with the noise in the matched-filter bandwidth B ~ 1 / tau_compressed.
"""

from __future__ import annotations

import numpy as np

from .config import C, K_BOLTZMANN, T0, Design


def db(x):
    return 10.0 * np.log10(x)


def undb(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def antenna_gain(beam_az_deg: float, beam_el_deg: float) -> float:
    """Directivity-based gain, G ~ 26,000 / (theta_az theta_el) in degrees.

    The rule of thumb (Skolnik, Introduction to Radar Systems) corresponds to
    4 pi / (theta_az theta_el) in radians with ~63 % aperture efficiency.
    """
    return 26_000.0 / (beam_az_deg * beam_el_deg)


def noise_power_w(design: Design) -> float:
    """Receiver noise power in the matched-filter bandwidth, k T0 F B."""
    return K_BOLTZMANN * T0 * float(undb(design.noise_figure_db)) * design.bandwidth_hz


def range_resolution_m(design: Design) -> float:
    return C / (2.0 * design.bandwidth_hz)


def compression_gain(design: Design) -> float:
    """Time-bandwidth product: transmitted pulse width over compressed width 1/B."""
    return design.pulse_width_s * design.bandwidth_hz


def atm_loss(range_m, loss_db_per_km: float):
    """Two-way atmospheric loss (linear) over slant range ``range_m``."""
    return undb(loss_db_per_km * np.asarray(range_m, dtype=float) / 1e3)


def snr_single_pulse(design: Design, rcs_m2, range_m, atm_db_per_km: float = 0.0):
    """Single-pulse SNR (linear) after the matched filter."""
    R = np.asarray(range_m, dtype=float)
    G = antenna_gain(design.beamwidth_az_deg, design.beamwidth_el_deg)
    lam = design.wavelength_m
    num = design.peak_power_w * design.pulse_width_s * G**2 * lam**2 * np.asarray(rcs_m2, dtype=float)
    den = ((4.0 * np.pi) ** 3 * R**4 * K_BOLTZMANN * T0 * undb(design.noise_figure_db)
           * undb(design.system_loss_db) * atm_loss(R, atm_db_per_km))
    return num / den


def rcs_from_snr(design: Design, snr, range_m, atm_db_per_km: float = 0.0):
    """Invert the radar equation: the RCS that gives ``snr`` at ``range_m``."""
    return np.asarray(snr, dtype=float) / snr_single_pulse(design, 1.0, range_m, atm_db_per_km)

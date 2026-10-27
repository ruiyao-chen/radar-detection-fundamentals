"""Airborne radar geometry over a smooth 4/3 Earth.

The radar is at height h above a sphere of radius a_e = 4/3 x 6371 km (the
standard-refraction effective Earth). For a ground point at grazing angle psi
and slant range R, the triangle centre-radar-point gives

    (a_e + h)^2 = a_e^2 + R^2 + 2 a_e R sin(psi)

and the ground (arc) range is a_e times the central angle beta, with
cos(beta) = (a_e^2 + (a_e + h)^2 - R^2) / (2 a_e (a_e + h)).
"""

from __future__ import annotations

import numpy as np

from .config import EARTH_RADIUS_M, K_EARTH

A_E = K_EARTH * EARTH_RADIUS_M


def slant_from_ground(ground_m, h_m: float, a_e: float = A_E):
    """Slant range (m) to a ground point at arc distance ``ground_m``."""
    beta = np.asarray(ground_m, dtype=float) / a_e
    # law of cosines rewritten with sin^2(beta/2) to avoid cancellation at small beta
    return np.sqrt(h_m**2 + 4.0 * a_e * (a_e + h_m) * np.sin(beta / 2.0) ** 2)


def grazing_from_slant(slant_m, h_m: float, a_e: float = A_E):
    """Grazing angle (rad) at the ground point, from the law of cosines above."""
    R = np.asarray(slant_m, dtype=float)
    s = (h_m * (2.0 * a_e + h_m) - R**2) / (2.0 * a_e * R)   # (a_e + h)^2 - a_e^2 expanded
    return np.arcsin(np.clip(s, -1.0, 1.0))


def slant_from_grazing(psi_rad, h_m: float, a_e: float = A_E):
    """Slant range (m) for a grazing angle, the positive root of the quadratic in R."""
    s = np.sin(np.asarray(psi_rad, dtype=float))
    return -a_e * s + np.sqrt((a_e * s) ** 2 + h_m**2 + 2.0 * a_e * h_m)


def ground_from_slant(slant_m, h_m: float, a_e: float = A_E):
    """Arc ground range (m) for a slant range."""
    R = np.asarray(slant_m, dtype=float)
    psi = grazing_from_slant(R, h_m, a_e)
    # central angle from its sine and cosine (atan2 keeps precision at small angles)
    sin_b = R * np.cos(psi) / (a_e + h_m)
    cos_b = (a_e + R * np.sin(psi)) / (a_e + h_m)
    return a_e * np.arctan2(sin_b, cos_b)


def horizon_m(h_m: float, a_e: float = A_E) -> float:
    """Slant range to the radar horizon (grazing angle zero)."""
    return float(np.sqrt(h_m**2 + 2.0 * a_e * h_m))

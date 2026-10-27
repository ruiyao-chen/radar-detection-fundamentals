"""Constants, the scenario (what the radar must do) and the baseline design.

The scenario is this repository's own: an X-band airborne surveillance radar
looking for moving ground vehicles. Every number marked "assumed" is a design
choice or a typical textbook value, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 20260427

C = 299_792_458.0            # speed of light, m/s
K_BOLTZMANN = 1.380649e-23   # J/K
T0 = 290.0                   # reference noise temperature, K
EARTH_RADIUS_M = 6_371_000.0
K_EARTH = 4.0 / 3.0          # standard-atmosphere refraction: effective Earth radius factor


@dataclass(frozen=True)
class Scenario:
    """What the radar must do (requirements) and the world it works in."""

    altitude_m: float = 6_000.0             # platform altitude
    platform_speed_ms: float = 100.0        # platform ground speed (360 km/h)
    ground_range_min_m: float = 10_000.0    # near edge of the search swath
    ground_range_max_m: float = 50_000.0    # far edge of the search swath
    sector_deg: float = 120.0               # scan sector, centred on broadside (azimuth 30-150 deg from the velocity vector)
    target_rcs_m2: float = 10.0             # ground vehicle, Swerling 1 (assumed)
    target_speed_min_ms: float = 2.0        # slowest radial speed to detect (7 km/h)
    target_speed_max_ms: float = 30.0       # fastest radial speed to measure unambiguously (108 km/h)
    pd_required: float = 0.80               # single-look Pd at the far edge, Swerling 1
    false_alarm_time_s: float = 60.0        # mean time between false alarms
    revisit_max_s: float = 4.0              # each point in the sector at least this often
    range_resolution_max_m: float = 15.0
    gamma_db: float = -15.0                 # constant-gamma land clutter (assumed; farmland / low hills)
    atm_loss_db_per_km: float = 0.02        # two-way, X band, clear air (assumed)


@dataclass(frozen=True)
class Design:
    """The radar's free parameters."""

    freq_hz: float = 10.0e9
    peak_power_w: float = 1_000.0
    pulse_width_s: float = 20e-6            # transmitted (uncompressed) pulse
    bandwidth_hz: float = 12e6              # chirp bandwidth -> compressed resolution c/2B
    prf_hz: float = 5_000.0
    beamwidth_az_deg: float = 1.2
    beamwidth_el_deg: float = 6.0
    scan_rate_deg_s: float = 40.0
    noise_figure_db: float = 4.0            # assumed
    system_loss_db: float = 6.0             # transmit, receive, processing, scan (assumed)
    max_duty: float = 0.15                  # transmitter duty-cycle limit (assumed)

    @property
    def wavelength_m(self) -> float:
        return C / self.freq_hz

    def with_(self, **kw) -> "Design":
        return replace(self, **kw)


SCENARIO = Scenario()
BASELINE = Design()

# Part 2: a window of range gates from the baseline radar, and the targets in it
PART2_RANGE0_M = 30_000.0         # slant range of the first gate
PART2_GATES = 200
PART2_TARGETS = (                 # (gate, RCS m^2): a truck, a small car, a person-sized return
    dict(gate=40, rcs_m2=10.0),
    dict(gate=95, rcs_m2=1.0),
    dict(gate=150, rcs_m2=0.1),
)
PART2_GATE_PFA = 1e-6             # false-alarm probability per range gate after M-of-N

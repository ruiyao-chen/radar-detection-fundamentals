"""Evaluate a design against the scenario, and sweep PRF x scan rate.

Detection is modelled with noncoherent integration of the pulses on target in
one scan, a Swerling 1 target and a square-law detector (exact Pd). Coherent
Doppler processing would add gain on top; it is not modelled, so the Pd here
is conservative for exo-clutter targets. Slow (endo-clutter) targets are
scored with the signal-to-clutter-plus-noise ratio, treating the clutter
residue like noise, which is optimistic for spiky clutter.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from . import clutter, detection, doppler, geometry, radar_eq
from .config import Design, Scenario

BROADSIDE = np.pi / 2.0


def evaluate(design: Design, sc: Scenario) -> dict:
    lam = design.wavelength_m
    az = np.radians(design.beamwidth_az_deg)
    el = np.radians(design.beamwidth_el_deg)
    r_near = float(geometry.slant_from_ground(sc.ground_range_min_m, sc.altitude_m))
    r_far = float(geometry.slant_from_ground(sc.ground_range_max_m, sc.altitude_m))
    psi_near = float(geometry.grazing_from_slant(r_near, sc.altitude_m))
    psi_far = float(geometry.grazing_from_slant(r_far, sc.altitude_m))
    dr = radar_eq.range_resolution_m(design)

    # timing
    revisit = sc.sector_deg / design.scan_rate_deg_s
    time_on_target = design.beamwidth_az_deg / design.scan_rate_deg_s
    n = max(int(np.floor(design.prf_hz * time_on_target)), 1)
    duty = design.pulse_width_s * design.prf_hz

    # detection, noise-limited, at the far edge
    pfa = detection.pfa_from_false_alarm_time(design.bandwidth_hz, sc.false_alarm_time_s)
    snr_far = float(radar_eq.snr_single_pulse(design, sc.target_rcs_m2, r_far, sc.atm_loss_db_per_km))
    pd_far = float(detection.pd_swerling1(snr_far, pfa, n))
    snr_req = detection.snr_required(sc.pd_required, pfa, n, swerling=1)
    pd_far_sw0 = float(detection.pd_swerling0(snr_far, pfa, n))
    pd_far_alb = float(detection.albersheim_pd(radar_eq.db(snr_far), pfa, n))

    # Doppler and range ambiguity (broadside, far edge: widest clutter band)
    spread = float(doppler.clutter_spread_hz(sc.platform_speed_ms, az, BROADSIDE, psi_far, lam))
    mdv = float(doppler.mdv_ms(sc.platform_speed_ms, az, BROADSIDE, psi_far))
    prf_min = float(doppler.min_prf_hz(sc.target_speed_max_ms, spread, lam))
    v_blind = float(doppler.blind_speed_ms(design.prf_hz, lam))
    r_ua = float(doppler.unambiguous_range_m(design.prf_hz))

    # clutter
    sig_c = {k: float(clutter.clutter_rcs_m2(r, p, az, el, dr, sc.gamma_db))
             for k, r, p in (("near", r_near, psi_near), ("far", r_far, psi_far))}
    cnr_far = float(radar_eq.snr_single_pulse(design, sig_c["far"], r_far, sc.atm_loss_db_per_km))
    scnr_far = snr_far / (1.0 + cnr_far)
    pd_far_slow = float(detection.pd_swerling1(scnr_far, pfa, n))

    m = dict(
        wavelength_m=lam, gain_db=float(radar_eq.db(radar_eq.antenna_gain(design.beamwidth_az_deg,
                                                                            design.beamwidth_el_deg))),
        range_resolution_m=dr, compression_gain_db=float(radar_eq.db(radar_eq.compression_gain(design))),
        duty=duty, avg_power_w=design.peak_power_w * duty,
        noise_power_dbm=float(radar_eq.db(radar_eq.noise_power_w(design)) + 30.0),
        slant_near_m=r_near, slant_far_m=r_far,
        grazing_near_deg=float(np.degrees(psi_near)), grazing_far_deg=float(np.degrees(psi_far)),
        cross_range_res_far_m=r_far * az,
        revisit_s=revisit, time_on_target_s=time_on_target, pulses_on_target=n,
        pfa=pfa, snr_far_db=float(radar_eq.db(snr_far)), snr_required_db=float(radar_eq.db(snr_req)),
        margin_db=float(radar_eq.db(snr_far / snr_req)),
        pd_far=pd_far, pd_far_swerling0=pd_far_sw0, pd_far_albersheim=pd_far_alb,
        clutter_spread_hz=spread, mdv_ms=mdv, prf_min_hz=prf_min, blind_speed_ms=v_blind,
        unambiguous_range_m=r_ua, range_ambiguities=int(doppler.n_range_ambiguities(r_far, design.prf_hz)),
        clutter_rcs_near_dbsm=float(radar_eq.db(sig_c["near"])),
        clutter_rcs_far_dbsm=float(radar_eq.db(sig_c["far"])),
        target_to_clutter_far_db=float(radar_eq.db(sc.target_rcs_m2 / sig_c["far"])),
        target_to_clutter_near_db=float(radar_eq.db(sc.target_rcs_m2 / sig_c["near"])),
        cnr_far_db=float(radar_eq.db(cnr_far)), pd_far_endo_clutter=pd_far_slow,
    )
    m["checks"] = checks(m, design, sc)
    m["feasible"] = all(c["pass"] for c in m["checks"] if c["required"])
    return m


def checks(m: dict, design: Design, sc: Scenario) -> list[dict]:
    def row(name, value, op, spec, required=True):
        ok = value <= spec if op == "<=" else value >= spec
        return dict(check=name, value=value, op=op, spec=spec, required=required, **{"pass": bool(ok)})

    return [
        row("Pd at far edge, Swerling 1", m["pd_far"], ">=", sc.pd_required),
        row("revisit time, s", m["revisit_s"], "<=", sc.revisit_max_s),
        row("range resolution, m", m["range_resolution_m"], "<=", sc.range_resolution_max_m),
        row("PRF for +/- v_max past the clutter, Hz", design.prf_hz, ">=", m["prf_min_hz"]),
        row("duty cycle", m["duty"], "<=", design.max_duty),
        row("MDV vs slowest target, m/s", m["mdv_ms"], "<=", sc.target_speed_min_ms),
        row("unambiguous range vs far edge, m", m["unambiguous_range_m"], ">=", m["slant_far_m"], required=False),
    ]


def trade_study(design: Design, sc: Scenario, prfs, scan_rates) -> pd.DataFrame:
    rows = []
    for prf in prfs:
        for w in scan_rates:
            m = evaluate(design.with_(prf_hz=float(prf), scan_rate_deg_s=float(w)), sc)
            c = {r["check"]: r["pass"] for r in m["checks"]}
            rows.append(dict(prf_hz=float(prf), scan_rate_deg_s=float(w), pulses=m["pulses_on_target"],
                             pd_far=m["pd_far"], revisit_s=m["revisit_s"], duty=m["duty"],
                             range_ambiguities=m["range_ambiguities"], feasible=m["feasible"],
                             fail_pd=not c["Pd at far edge, Swerling 1"], fail_revisit=not c["revisit time, s"],
                             fail_doppler=not c["PRF for +/- v_max past the clutter, Hz"],
                             fail_duty=not c["duty cycle"]))
    return pd.DataFrame(rows)


def design_table(design: Design) -> pd.DataFrame:
    return pd.DataFrame([dict(parameter=k, value=v) for k, v in asdict(design).items()])

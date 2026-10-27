#!/usr/bin/env python3
"""Part 1: size the airborne moving-target radar and run the trade study.

Writes results/part1/ (design, requirement checks, range budget, trade study,
Albersheim accuracy, summary.json) and figures/part1_*.png.

    python scripts/run_part1.py
    python scripts/run_part1.py --fast     # coarser trade grid (CI smoke run)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from pulseradar import clutter, design, detection, geometry, plotting, radar_eq  # noqa: E402
from pulseradar.config import BASELINE, FIGURES, RESULTS, SCENARIO  # noqa: E402

OUT = RESULTS / "part1"


def range_budget(d, sc, m) -> pd.DataFrame:
    """Per-pulse SNR, CNR and per-look Pd across the swath for the baseline."""
    ground = np.linspace(sc.ground_range_min_m, sc.ground_range_max_m, 41)
    R = geometry.slant_from_ground(ground, sc.altitude_m)
    psi = geometry.grazing_from_slant(R, sc.altitude_m)
    snr = radar_eq.snr_single_pulse(d, sc.target_rcs_m2, R, sc.atm_loss_db_per_km)
    sig_c = clutter.clutter_rcs_m2(R, psi, np.radians(d.beamwidth_az_deg), np.radians(d.beamwidth_el_deg),
                                   radar_eq.range_resolution_m(d), sc.gamma_db)
    cnr = radar_eq.snr_single_pulse(d, sig_c, R, sc.atm_loss_db_per_km)
    n, pfa = m["pulses_on_target"], m["pfa"]
    return pd.DataFrame(dict(
        ground_km=ground / 1e3, slant_km=R / 1e3, grazing_deg=np.degrees(psi),
        snr_db=radar_eq.db(snr), cnr_db=radar_eq.db(cnr), tcr_db=radar_eq.db(sc.target_rcs_m2 / sig_c),
        pd_sw1=detection.pd_swerling1(snr, pfa, n), pd_sw0=detection.pd_swerling0(snr, pfa, n),
        pd_albersheim_sw0=detection.albersheim_pd(radar_eq.db(snr), pfa, n),
        pd_endo=detection.pd_swerling1(snr / (1.0 + cnr), pfa, n)))


def albersheim_accuracy() -> pd.DataFrame:
    rows = []
    for pfa in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        for n in (1, 10, 100):
            for pd_ in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
                exact = radar_eq.db(detection.snr_required(pd_, pfa, n, 0))
                alb = float(detection.albersheim_snr_db(pd_, pfa, n))
                rows.append(dict(pfa=pfa, n=n, pd=pd_, exact_db=float(exact), albersheim_db=alb,
                                 error_db=alb - float(exact)))
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    d, sc = BASELINE, SCENARIO
    m = design.evaluate(d, sc)
    print("baseline design:", json.dumps(asdict(d)))
    for k, v in m.items():
        if k not in ("checks",):
            print(f"  {k:28} {v}")
    checks = pd.DataFrame(m["checks"])
    print("\n" + checks.to_string(index=False))
    checks.to_csv(OUT / "checks.csv", index=False)
    design.design_table(d).to_csv(OUT / "design.csv", index=False)

    budget = range_budget(d, sc, m)
    budget.round(4).to_csv(OUT / "range_budget.csv", index=False)

    prfs = np.arange(1_000, 12_001, 1_000 if args.fast else 250)
    rates = np.arange(10, 161, 10 if args.fast else 2.5)
    trade = design.trade_study(d, sc, prfs, rates)
    trade.round(4).to_csv(OUT / "trade_study.csv", index=False)
    feas = trade[trade.feasible]
    print(f"\ntrade study: {len(trade)} designs, {len(feas)} feasible; "
          f"PRF {feas.prf_hz.min():.0f}-{feas.prf_hz.max():.0f} Hz, scan {feas.scan_rate_deg_s.min():.1f}-"
          f"{feas.scan_rate_deg_s.max():.1f} deg/s" if len(feas) else "no feasible designs")

    alb = albersheim_accuracy()
    alb.round(4).to_csv(OUT / "albersheim_accuracy.csv", index=False)
    mid = alb[(alb.pd >= 0.5)]
    print(f"\nAlbersheim - exact: Pd 0.5-0.9 max |err| {mid.error_db.abs().max():.2f} dB; "
          f"all Pd max |err| {alb.error_db.abs().max():.2f} dB at "
          f"{alb.loc[alb.error_db.abs().idxmax(), ['pfa', 'n', 'pd']].to_dict()}")

    figs = {
        "part1_range_budget": plotting.plot_range_budget(budget, m["snr_required_db"], sc.pd_required),
        "part1_trade_study": plotting.plot_trade(trade, (d.prf_hz, d.scan_rate_deg_s)),
        "part1_albersheim": plotting.plot_albersheim_error(alb),
    }
    for name, f in figs.items():
        f.savefig(FIGURES / f"{name}.png", dpi=150)

    summary = dict(design=asdict(d), scenario=asdict(sc),
                   metrics={k: v for k, v in m.items() if k != "checks"}, checks=m["checks"],
                   trade=dict(n_designs=int(len(trade)), n_feasible=int(len(feas)),
                              fail_counts={c: int(trade[c].sum()) for c in trade.columns if c.startswith("fail_")}),
                   albersheim=dict(max_abs_err_db_pd_0p5_0p9=float(mid.error_db.abs().max()),
                                   max_abs_err_db_all=float(alb.error_db.abs().max()),
                                   worst=alb.loc[alb.error_db.abs().idxmax()].to_dict()),
                   budget_endpoints=budget.iloc[[0, -1]].round(4).to_dict(orient="records"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    info = dict(script="run_part1.py", args=vars(args), utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                runtime_s=round(time.time() - t0, 1), python=platform.python_version(),
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__))
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"\ndone in {time.time() - t0:.0f} s -> results/part1, figures/part1_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

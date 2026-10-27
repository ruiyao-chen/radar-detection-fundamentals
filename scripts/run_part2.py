#!/usr/bin/env python3
"""Part 2: threshold detection in the baseline radar's range-gate x pulse matrix.

A 200-gate window starting at 30 km slant range, one dwell of the baseline
design (its pulses on target), three echoes whose SNR comes from the radar
equation for 10, 1 and 0.1 m^2. Noise is estimated from transmitter-off data,
the single-pulse threshold and M are chosen for a gate-level Pfa of 1e-6, and
the detected gates' SNR and RCS are estimated. Monte Carlo checks: Pfa when the
noise level is estimated from K samples, Pd vs SNR, M-of-N, and CA-CFAR on a
noise floor that steps up with range.

Writes results/part2/ and figures/part2_*.png.

    python scripts/run_part2.py
    python scripts/run_part2.py --fast     # fewer Monte Carlo trials (CI smoke run)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy import optimize  # noqa: E402

from pulseradar import design, detection, echoes as E, plotting, radar_eq  # noqa: E402
from pulseradar.config import (BASELINE, FIGURES, PART2_GATE_PFA, PART2_GATES, PART2_RANGE0_M,  # noqa: E402
                               PART2_TARGETS, RESULTS, SCENARIO, SEED)

OUT = RESULTS / "part2"


def choose_binary_integration(n: int, gate_pfa: float, pd_goal: float = 0.9) -> pd.DataFrame:
    """For each single-pulse Pfa, the smallest M that meets the gate Pfa, and the SNR it then needs."""
    rows = []
    for p1 in (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4):
        m = next(k for k in range(1, n + 1) if detection.m_of_n(p1, k, n) <= gate_pfa)
        # single-pulse Pd needed so that M of n crossings happen with probability pd_goal
        lo, hi = 1e-6, 1 - 1e-12
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if detection.m_of_n(mid, m, n) < pd_goal else (lo, mid)
        pd1 = hi
        f = lambda s_db: float(detection.pd_single_pulse(10 ** (s_db / 10), p1)) - pd1  # noqa: E731
        snr_db = float(optimize.brentq(f, -30, 40))
        rows.append(dict(pfa_single=p1, m=m, n=n, gate_pfa=float(detection.m_of_n(p1, m, n)),
                         pd_single_needed=pd1, snr_needed_db=snr_db))
    df = pd.DataFrame(rows)
    df["noncoherent_snr_needed_db"] = radar_eq.db(detection.snr_required(pd_goal, gate_pfa, n, 0))
    df["binary_loss_db"] = df.snr_needed_db - df.noncoherent_snr_needed_db
    return df


def pfa_vs_k(rng, pfa: float, ks, trials: int, test_samples: int) -> pd.DataFrame:
    rows = []
    L = np.log(1.0 / pfa)
    for K in ks:
        hits = 0
        for _ in range(trials):
            sig = E.sigma_ml(E.simulate(1, K, sigma=1.0, rng=rng))
            hits += int(E.crossings(E.simulate(1, test_samples, sigma=1.0, rng=rng), E.threshold(sig, pfa)).sum())
        rows.append(dict(k_samples=K, design=pfa, measured=hits / (trials * test_samples),
                         theory=(1.0 + L / K) ** (-K), n_tested=trials * test_samples))
    return pd.DataFrame(rows)


def pd_vs_snr(rng, pfa: float, snrs_db, trials: int) -> pd.DataFrame:
    rows = []
    for s_db in snrs_db:
        s = 10 ** (s_db / 10)
        r = E.simulate(1, trials, [E.Echo(0, s)], rng=rng)
        rows.append(dict(snr_db=s_db, pd_mc=float(E.crossings(r, E.threshold(1.0, pfa)).mean()),
                         pd_exact=float(detection.pd_single_pulse(s, pfa)),
                         pd_albersheim=float(detection.albersheim_pd(s_db, pfa, 1)), trials=trials))
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    d, sc = BASELINE, SCENARIO
    m = design.evaluate(d, sc)
    n = m["pulses_on_target"]
    dr = radar_eq.range_resolution_m(d)
    ranges = PART2_RANGE0_M + dr * np.arange(PART2_GATES)

    # --- 1. choose the single-pulse Pfa and M --------------------------------- #
    bi = choose_binary_integration(n, PART2_GATE_PFA)
    best = bi.loc[bi.snr_needed_db.idxmin()]
    pfa1, M = float(best.pfa_single), int(best.m)
    print(f"dwell: {n} pulses; gate Pfa {PART2_GATE_PFA:g}\n{bi.round(4).to_string(index=False)}")
    print(f"-> single-pulse Pfa {pfa1:g}, M = {M} of {n}")
    bi.round(6).to_csv(OUT / "binary_integration.csv", index=False)

    # --- 2. one dwell: noise estimate, threshold, M-of-N, estimates ----------- #
    truth = []
    for t in PART2_TARGETS:
        s = float(radar_eq.snr_single_pulse(d, t["rcs_m2"], ranges[t["gate"]], sc.atm_loss_db_per_km))
        truth.append(dict(gate=t["gate"], rcs_m2=t["rcs_m2"], range_m=float(ranges[t["gate"]]), snr_db=float(radar_eq.db(s)),
                          pd_gate_theory=float(detection.m_of_n(detection.pd_single_pulse(s, pfa1), M, n))))
    quiet = E.simulate(PART2_GATES, n, sigma=1.0, rng=rng)                        # transmitter off
    sig_mean, sig_ml = E.sigma_from_mean(quiet), E.sigma_ml(quiet)
    r = E.simulate(PART2_GATES, n, [E.Echo(t["gate"], 10 ** (t["snr_db"] / 10)) for t in truth], sigma=1.0, rng=rng)
    hits = E.crossings(r, E.threshold(sig_ml, pfa1))
    declared = np.flatnonzero(E.m_of_n_detect(hits, M))
    rows = []
    for g in declared:
        s_hat = E.estimate_snr(r[g], sig_ml)
        tr = next((t for t in truth if t["gate"] == g), None)
        rows.append(dict(gate=int(g), range_m=float(ranges[g]), crossings=int(hits[g].sum()),
                         snr_hat_db=float(radar_eq.db(s_hat)) if s_hat > 0 else None,
                         rcs_hat_m2=float(radar_eq.rcs_from_snr(d, s_hat, ranges[g], sc.atm_loss_db_per_km)),
                         true_rcs_m2=tr["rcs_m2"] if tr else None, true_snr_db=tr["snr_db"] if tr else None))
    det = pd.DataFrame(rows)
    print(f"\nnoise sigma (true 1): mean-based {sig_mean:.4f}, ML {sig_ml:.4f}")
    print("truth:\n" + pd.DataFrame(truth).round(4).to_string(index=False))
    print("declared:\n" + det.round(4).to_string(index=False))
    det.round(6).to_csv(OUT / "detections.csv", index=False)
    pd.DataFrame(truth).round(6).to_csv(OUT / "truth.csv", index=False)

    # --- 3. Monte Carlo checks --------------------------------------------- #
    trials = 200 if args.fast else 2000
    pk = pfa_vs_k(rng, 1e-4, [10, 30, 100, 300, 1000, 10000], trials, 5000)
    print("\nPfa with sigma estimated from K samples:\n" + pk.to_string(index=False))
    pk.to_csv(OUT / "pfa_vs_k.csv", index=False)

    pdv = pd_vs_snr(rng, 1e-3, np.arange(-2.0, 16.1, 1.0), 20_000 if args.fast else 200_000)
    pdv["abs_err_mc"] = (pdv.pd_mc - pdv.pd_exact).abs()
    print("\nPd vs SNR (single pulse, Pfa 1e-3):\n" + pdv.round(4).to_string(index=False))
    pdv.round(6).to_csv(OUT / "pd_vs_snr.csv", index=False)

    dwells = 2000 if args.fast else 20000
    mn = []
    for s_db in (-4.0, -3.0, -2.0, -1.0):
        s = 10 ** (s_db / 10)
        count = 0
        for _ in range(dwells // 1000):
            rr = E.simulate(1000, n, [E.Echo(g, s) for g in range(1000)], rng=rng)
            count += int(E.m_of_n_detect(E.crossings(rr, E.threshold(1.0, pfa1)), M).sum())
        mn.append(dict(snr_db=s_db, pd_gate_mc=count / dwells,
                       pd_gate_theory=float(detection.m_of_n(detection.pd_single_pulse(s, pfa1), M, n)), dwells=dwells))
    mn = pd.DataFrame(mn)
    print("\nM-of-N gate Pd:\n" + mn.round(4).to_string(index=False))
    mn.to_csv(OUT / "m_of_n.csv", index=False)

    # --- 4. CA-CFAR on a stepped noise floor -------------------------------- #
    pfa_c, G = 1e-3, 400
    sigma = np.where(np.arange(G) < 200, 1.0, np.sqrt(10.0))
    rc = E.simulate(G, 500 if args.fast else 5000, sigma=sigma, rng=rng)
    sig_quiet = E.sigma_ml(rc[:200])
    rate_fixed = E.crossings(rc, E.threshold(sig_quiet, pfa_c)).mean(axis=1)
    cdet, _ = E.ca_cfar(rc, pfa_c, guard=2, train=16)
    rate_cfar = cdet.mean(axis=1)
    cf = pd.DataFrame(dict(gate=np.arange(G), noise_db=20 * np.log10(sigma), rate_fixed=rate_fixed, rate_cfar=rate_cfar))
    cf.to_csv(OUT / "cfar_step.csv", index=False)
    regions = {k: dict(fixed=float(rate_fixed[s].mean()), cfar=float(rate_cfar[s].mean()))
               for k, s in (("quiet_gates_20_180", slice(20, 180)), ("loud_gates_220_380", slice(220, 380)),
                            ("edge_gates_190_210", slice(190, 210)))}
    print("\nCFAR step (design Pfa 1e-3):", json.dumps(regions, indent=1))

    figs = {
        "part2_matrix": plotting.plot_matrix(r, hits, M, sig_ml, truth, declared),
        "part2_pfa_pd": plotting.plot_pfa_pd(pk, pdv),
        "part2_cfar": plotting.plot_cfar(rate_fixed, rate_cfar, 20 * np.log10(sigma), pfa_c),
    }
    for name, f in figs.items():
        f.savefig(FIGURES / f"{name}.png", dpi=150)

    summary = dict(pulses=n, range_resolution_m=dr, gate_pfa=PART2_GATE_PFA, pfa_single=pfa1, m=M,
                   binary_integration=bi.to_dict(orient="records"), sigma_mean=sig_mean, sigma_ml=sig_ml,
                   truth=truth, declared=det.to_dict(orient="records"),
                   pfa_vs_k=pk.to_dict(orient="records"), pd_vs_snr_max_abs_err=float(pdv.abs_err_mc.max()),
                   m_of_n=mn.to_dict(orient="records"), cfar_step=regions)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    info = dict(script="run_part2.py", args=vars(args), seed=SEED,
                utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                runtime_s=round(time.time() - t0, 1), python=platform.python_version(),
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__))
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"\ndone in {time.time() - t0:.0f} s -> results/part2, figures/part2_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

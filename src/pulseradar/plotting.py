"""Report figures (static PNGs for the README).

Colours are the first categorical slots of a colour-blind-validated palette
(blue, orange, aqua, violet), always paired with a marker, line style or label
so no category is told apart by colour alone. One y-axis per panel.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURFACE, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b7b1"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    })


# --------------------------------------------------------------------------- #
# Part 1
# --------------------------------------------------------------------------- #
def plot_range_budget(df: pd.DataFrame, snr_req_db: float, pd_req: float):
    """SNR and clutter-to-noise per pulse, and Pd per look, against ground range."""
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = df.ground_km
    a.plot(x, df.snr_db, color=BLUE, label="target SNR per pulse (10 m²)")
    a.plot(x, df.cnr_db, color=ORANGE, ls="--", label="clutter-to-noise per pulse")
    a.axhline(snr_req_db, color=INK2, lw=1, ls=":")
    a.text(x.iloc[-1], snr_req_db + 0.6, f"needed for Pd {pd_req:.1f}, Swerling 1", ha="right", color=INK2)
    a.set(xlabel="ground range (km)", ylabel="dB", title="Per-pulse budget")
    a.legend(loc="upper right")
    b.plot(x, df.pd_sw1, color=BLUE, label="exact, Swerling 1 (design case)")
    b.plot(x, df.pd_sw0, color=AQUA, ls="--", label="exact, steady target")
    b.plot(x, df.pd_endo, color=ORANGE, ls="-.", label="slow mover inside clutter, Swerling 1")
    b.axhline(pd_req, color=INK2, lw=1, ls=":")
    b.set(xlabel="ground range (km)", ylabel="Pd per look", ylim=(-0.02, 1.02), title="Detection per scan")
    b.legend(loc="center left")
    fig.tight_layout()
    return fig


def plot_trade(df: pd.DataFrame, baseline: tuple[float, float]):
    """PRF x scan-rate map coloured by the first requirement each design fails."""
    style()
    order = ["feasible", "Pd too low", "revisit too slow", "PRF below Doppler need", "duty cycle over limit"]
    colors = [AQUA, ORANGE, VIOLET, BLUE, MUTED]

    def label(r):     # hardware and Doppler limits first, then timing, then Pd
        if r.feasible:
            return 0
        if r.fail_doppler:
            return 3
        if r.fail_duty:
            return 4
        if r.fail_revisit:
            return 2
        return 1

    prfs = np.sort(df.prf_hz.unique())
    rates = np.sort(df.scan_rate_deg_s.unique())
    grid = np.zeros((len(rates), len(prfs)), dtype=int)
    for _, r in df.iterrows():
        grid[np.searchsorted(rates, r.scan_rate_deg_s), np.searchsorted(prfs, r.prf_hz)] = label(r)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    dp, dr = np.diff(prfs).mean(), np.diff(rates).mean()
    ax.imshow(grid, origin="lower", cmap=ListedColormap(colors), vmin=-0.5, vmax=len(order) - 0.5, aspect="auto",
              extent=[(prfs[0] - dp / 2) / 1e3, (prfs[-1] + dp / 2) / 1e3, rates[0] - dr / 2, rates[-1] + dr / 2],
              interpolation="nearest")
    ax.grid(False)
    ax.plot(baseline[0] / 1e3, baseline[1], marker="*", ms=14, color=INK, mec=SURFACE, ls="none")
    ax.annotate("baseline", (baseline[0] / 1e3, baseline[1]), xytext=(8, 6), textcoords="offset points")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, order, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set(xlabel="PRF (kHz)", ylabel="scan rate (deg/s)", title="Which requirement fails first")
    fig.tight_layout()
    return fig


def plot_albersheim_error(df: pd.DataFrame):
    """Albersheim required SNR minus the exact square-law value, against Pd."""
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for (pfa, color, mk) in ((1e-3, ORANGE, "o"), (1e-6, BLUE, "s"), (1e-7, AQUA, "^")):
        for n, ls in ((1, "-"), (100, "--")):
            d = df[(df.pfa == pfa) & (df.n == n)]
            ax.plot(d.pd, d.error_db, color=color, ls=ls, marker=mk, ms=4, label=f"Pfa {pfa:g}, n = {n}")
    ax.axhspan(-0.2, 0.2, color=GRID, zorder=0)
    ax.set(xlabel="Pd", ylabel="Albersheim − exact (dB)", title="Albersheim's equation vs the exact SNR")
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Part 2
# --------------------------------------------------------------------------- #
def plot_matrix(r: np.ndarray, hits: np.ndarray, m: int, sigma: float, truth: list[dict], detected_gates):
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1]))
    im = a.imshow(20 * np.log10(r / sigma), origin="lower", aspect="auto", cmap="gray", vmin=-10, vmax=20,
                  interpolation="nearest")
    a.grid(False)
    a.set(xlabel="pulse number (slow time)", ylabel="range gate (fast time)", title="Detector output, dB re noise")
    fig.colorbar(im, ax=a, fraction=0.04)
    count = hits.sum(axis=1)
    g = np.arange(len(count))
    b.bar(g, count, width=1.0, color=MUTED)
    b.bar(g[detected_gates], count[detected_gates], width=2.0, color=BLUE, label="declared target")
    b.axhline(m, color=ORANGE, ls="--", lw=1.4, label=f"M = {m}")
    for t in truth:
        tg = t["gate"]
        txt = f"{t['rcs_m2']:g} m²" + ("" if tg in set(detected_gates) else " (missed)")
        b.annotate(txt, (tg, count[tg]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    b.set(xlabel="range gate", ylabel="threshold crossings over the pulses", title="M-of-N decision per gate")
    b.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_pfa_pd(pfa_df: pd.DataFrame, pd_df: pd.DataFrame):
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    a.semilogx(pfa_df.k_samples, pfa_df.measured / pfa_df.design, color=BLUE, marker="o", ls="none",
               label="measured (Monte Carlo)")
    a.semilogx(pfa_df.k_samples, pfa_df.theory / pfa_df.design, color=ORANGE, ls="--", label="(1 + ln(1/Pfa)/K)^−K")
    a.axhline(1.0, color=INK2, lw=1, ls=":")
    a.set(xlabel="noise samples used to estimate σ (K)", ylabel="actual Pfa / design Pfa",
          title="Threshold from an estimated noise level")
    a.legend()
    b.plot(pd_df.snr_db, pd_df.pd_mc, color=BLUE, marker="o", ls="none", label="Monte Carlo")
    b.plot(pd_df.snr_db, pd_df.pd_exact, color=INK2, lw=1.4, label="exact (Marcum Q)")
    b.plot(pd_df.snr_db, pd_df.pd_albersheim, color=ORANGE, ls="--", label="Albersheim, solved for Pd")
    b.set(xlabel="single-pulse SNR (dB)", ylabel="Pd, one pulse", title="Pd vs SNR, Pfa = 1e-3")
    b.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_cfar(rate_fixed: np.ndarray, rate_cfar: np.ndarray, sigma_db: np.ndarray, pfa: float):
    style()
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2]))
    g = np.arange(len(sigma_db))
    a.plot(g, sigma_db, color=INK2)
    a.set(ylabel="noise floor (dB)", title="A noise floor that steps up 10 dB with range")
    blk = 10                                                   # average over blocks of 10 gates
    gb = g.reshape(-1, blk).mean(axis=1)
    fb, cb = rate_fixed.reshape(-1, blk).mean(axis=1), rate_cfar.reshape(-1, blk).mean(axis=1)
    b.semilogy(gb, fb, color=ORANGE, marker="o", ms=3, label="fixed threshold (set on the quiet gates)")
    b.semilogy(gb, np.maximum(cb, 1e-5), color=BLUE, marker="s", ms=3, ls="--", label="cell-averaging CFAR")
    b.axhline(pfa, color=INK2, lw=1, ls=":")
    b.text(g[-1], pfa * 0.6, "design Pfa", color=INK2, ha="right", va="top")
    b.set(xlabel="range gate", ylabel="false alarms per sample\n(10-gate blocks)", ylim=(1e-5, 1.5))
    b.legend(loc="upper left")
    fig.tight_layout()
    return fig

"""
Generate figures from the CSV outputs of main.py.

Reads:
    results/trajectory__<method>__sigma-<case>.csv   (per-hour trajectories)
    results/fig8_costs.csv                           (long-format per-realization costs)

Writes:
    figures/trajectory__<method>__sigma-<case>.png   (dark-theme 4-panel dashboard)
    figures/fig8_histograms.png                       (Fig 8 replica, det+stoch+perfect)

Usage
-----
    python generate_figures.py
    python generate_figures.py --trajectory   # only per-method dashboards
    python generate_figures.py --fig8         # only the histograms
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


PROJ_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJ_ROOT, "results")
FIGURES_DIR  = os.path.join(PROJ_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme palette  (matches previous full-info plot style)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = dict(
    GRID="#2a2a40", ACCENT="#f0c040", BLUE="#4fc3f7", GREEN="#81c784",
    RED="#ef9a9a", PURPLE="#ce93d8", TEXT="#e0e0e0",
    BG_FIG="#0f0f1a", BG_AX="#16162a",
)


def _style_ax(ax, ylabel, title, xlim):
    p = PALETTE
    ax.set_facecolor(p["BG_AX"])
    ax.tick_params(colors=p["TEXT"])
    ax.set_ylabel(ylabel, color=p["TEXT"], fontsize=9)
    ax.set_title(title,  color=p["ACCENT"], fontsize=10, pad=4)
    for sp in ax.spines.values():
        sp.set_color(p["GRID"]); sp.set_linewidth(0.6)
    ax.grid(True, color=p["GRID"], lw=0.5)
    ax.set_xlim(0, xlim)


# ─────────────────────────────────────────────────────────────────────────────
# Per-controller trajectory dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_trajectory(csv_path: str, title: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    T = len(df)
    hours = df["t"].to_numpy()
    p = PALETTE

    E_full = np.r_[df["E_kWh"].to_numpy(), df["E_kWh"].iloc[-1]]  # pad final SoC
    P = df["P_kW"].to_numpy()
    F = df["F_kW"].to_numpy()
    d = df["d_kW"].to_numpy()
    L = df["L_kW"].to_numpy()
    pi_e = df["pi_e_per_kWh"].to_numpy()
    pi_f = df["pi_f_per_kW"].to_numpy()
    alpha = df["alpha"].to_numpy()

    energy = float(df["cost_e_usd"].sum())
    fr_rev = float(df["rev_fr_usd"].sum())
    peak   = float(d.max())
    # Rough demand charge (one billing period assumed, no intra-window reset)
    pi_D_est = 20.0
    demand   = pi_D_est * peak
    net      = energy - fr_rev + demand

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(p["BG_FIG"])
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    # Panel 1 — load vs grid draw
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(hours, L, color=p["BLUE"],  lw=1.2, label="Load $L_t$")
    ax1.plot(hours, d, color=p["GREEN"], lw=1.2, ls="--", label="Grid draw $d_t$")
    ax1.axhline(peak, color=p["RED"], lw=0.9, ls=":", label=f"Peak D={peak:.0f} kW")
    _style_ax(ax1, "kW", "Load vs. Grid Draw", T - 1)
    ax1.legend(fontsize=8, facecolor=p["BG_AX"], labelcolor=p["TEXT"], loc="upper right")

    # Panel 2 — battery power & FR
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(hours, P, 0, where=P > 0, color=p["RED"],  alpha=0.6, label="Discharge")
    ax2.fill_between(hours, P, 0, where=P < 0, color=p["BLUE"], alpha=0.6, label="Charge")
    ax2.plot(hours, F, color=p["PURPLE"], lw=1.2, label="FR capacity $F_t$")
    _style_ax(ax2, "kW", "Battery Power & FR Capacity", T - 1)
    ax2.legend(fontsize=8, facecolor=p["BG_AX"], labelcolor=p["TEXT"], loc="upper right")

    # Panel 3 — SoC
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(np.arange(T + 1), E_full, color=p["ACCENT"], alpha=0.25)
    ax3.plot(np.arange(T + 1), E_full, color=p["ACCENT"], lw=1.3, label="SoC $E_t$")
    ax3.axhline(500.0, color=p["RED"],  lw=0.8, ls="--", label="Max 500 kWh")
    ax3.axhline(250.0, color=p["GRID"], lw=0.8, ls=":",  label="50%")
    _style_ax(ax3, "kWh", "Battery State of Charge", T)
    ax3.legend(fontsize=8, facecolor=p["BG_AX"], labelcolor=p["TEXT"])

    # Panel 4 — prices + ISO signal
    ax4  = fig.add_subplot(gs[3])
    ax4b = ax4.twinx()
    ax4.plot(hours, pi_e * 1000, color=p["GREEN"],  lw=1.0, label="Elec. price [$/MWh]")
    ax4.plot(hours, pi_f * 1000, color=p["PURPLE"], lw=1.0, label="FR price [$/MW-h]")
    ax4b.fill_between(hours, alpha, 0, color=p["BLUE"], alpha=0.25, label=r"$\alpha_t$")
    ax4b.set_ylabel(r"$\alpha$ [-]", color=p["BLUE"], fontsize=9)
    ax4b.tick_params(colors=p["BLUE"])
    for sp in ax4b.spines.values():
        sp.set_color(p["GRID"]); sp.set_linewidth(0.6)
    _style_ax(ax4, "$ / MWh", "Market Prices & ISO FR Signal", T - 1)
    ax4.set_xlabel("Hour", color=p["TEXT"], fontsize=9)
    lines1, labs1 = ax4.get_legend_handles_labels()
    lines2, labs2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labs1 + labs2,
               fontsize=8, facecolor=p["BG_AX"], labelcolor=p["TEXT"], loc="upper right")

    # Day dividers
    for ax in (ax1, ax2, ax3, ax4):
        for d_ in range(1, T // 24 + 1):
            ax.axvline(d_ * 24, color=p["GRID"], lw=0.6, ls="--")

    fig.suptitle(
        f"{title}\n"
        f"Net ${net:,.0f}  (energy ${energy:,.0f}  "
        f"− FR ${fr_rev:,.0f}  + demand ${demand:,.0f})",
        color=p["TEXT"], fontsize=11, y=0.99,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def render_all_trajectories() -> None:
    """Render every trajectory CSV found in results/ into a dashboard PNG."""
    pattern = os.path.join(RESULTS_DIR, "trajectory__*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"  no trajectory CSVs found in {pattern}")
        return

    for path in paths:
        base = os.path.basename(path)[:-4]  # strip .csv
        # base = "trajectory__<method>__sigma-<case>"
        m = re.match(r"trajectory__(?P<method>[^_]+(?:_[^_]+)*?)__sigma-(?P<case>[^.]+)", base)
        if m:
            method, case = m.group("method"), m.group("case")
            title = f"{method.replace('_', ' ').title()} MPC  |  σ={case}"
        else:
            title = base

        out = os.path.join(FIGURES_DIR, f"{base}.png")
        plot_trajectory(path, title, out)
        print(f"  {os.path.relpath(path, PROJ_ROOT)}  ->  {os.path.relpath(out, PROJ_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 replica — histograms of monthly cost by method × σ
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig8(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    methods = ["deterministic", "stochastic", "perfect_info"]
    cases   = sorted(df["sigma_case"].unique())
    colors  = {"deterministic": "indianred",
               "stochastic":    "steelblue",
               "perfect_info":  "seagreen"}
    titles  = {"deterministic": "Deterministic",
               "stochastic":    "Stochastic",
               "perfect_info":  "Perfect Information"}

    n_rows, n_cols = len(cases), len(methods)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows),
                            sharey="row")
    if n_rows == 1:
        axs = np.array([axs])   # keep 2-D indexing

    for r, case in enumerate(cases):
        row_df = df[df["sigma_case"] == case]
        lo = row_df["phi"].min()
        hi = row_df["phi"].max()
        pad = 0.05 * (hi - lo if hi > lo else abs(hi) + 1.0)
        xlim = (lo - pad, hi + pad)
        bins = np.linspace(*xlim, 25)

        for c, method in enumerate(methods):
            ax = axs[r, c]
            vals = row_df[row_df["method"] == method]["phi"].to_numpy()
            ax.hist(vals, bins=bins, density=True, alpha=0.75,
                    color=colors[method], edgecolor="black", linewidth=0.5)
            ax.axvline(vals.mean(), color="black", lw=1.5, ls="--",
                       label=f"Mean ${vals.mean():,.0f}")
            ax.set_title(f"{titles[method]}  (σ={case})", fontsize=11)
            ax.set_xlim(*xlim)
            ax.set_xlabel("Monthly total cost ($)", fontsize=9)
            if c == 0:
                ax.set_ylabel("Density", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Fig. 8 replica  —  monthly cost distribution under {df['realization'].max()+1} "
        f"evaluation realizations  (rows = σ case, cols = method)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def render_fig8() -> None:
    path = os.path.join(RESULTS_DIR, "fig8_costs.csv")
    if not os.path.exists(path):
        print(f"  no Fig 8 CSV found at {path}")
        return
    out = os.path.join(FIGURES_DIR, "fig8_histograms.png")
    plot_fig8(path, out)
    print(f"  {os.path.relpath(path, PROJ_ROOT)}  ->  {os.path.relpath(out, PROJ_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", action="store_true")
    ap.add_argument("--fig8",       action="store_true")
    args = ap.parse_args()

    if not args.trajectory and not args.fig8:
        args.trajectory = True
        args.fig8 = True

    if args.trajectory:
        print("Rendering trajectory dashboards …")
        render_all_trajectories()

    if args.fig8:
        print("\nRendering Fig 8 histograms …")
        render_fig8()


if __name__ == "__main__":
    main()

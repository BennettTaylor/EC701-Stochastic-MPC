"""
Replicate Fig. 8 of Kumar et al. — monthly-cost distribution for
*perfect-information* and *stochastic* MPC with a 7-day horizon,
under both undiscounted (σ=1) and discounted (σ=M/N) demand charges.
Deterministic MPC is intentionally omitted per the user's request.

Procedure
---------
1. Fit scenario generators on the full NYISO 2023 year  (done in
   stochatic_mpc.SeasonalLedoitWolfEstimator etc; reused from stoch_mpc.py).
2. For each σ ∈ {1, M/N}:
   a. Run stochastic MPC ONCE over a `MONTH_HOURS` billing period using the
      real NYISO realization for state updates.  This yields ONE commitment
      trajectory {P_stoch_t, F_stoch_t} for that σ.
   b. Draw |Ξ| = N_REALIZATIONS joint realizations of (L, π^e, π^f, α) over
      the same billing period from the fitted distributions.
   c. For every ξ ∈ Ξ:
        • Compute Φ_stoch(ξ) by plugging ξ into the paper's cost formula
          (IV.11) with the FIXED stochastic commitment — matches the paper's
          approach ("use the commitment policies PM, FM obtained under the
          stochastic MPC scheme and the actual realizations ... to compute
          the accumulated profit").
        • Run perfect-information MPC on ξ itself (RH LP with ξ treated as
          known future) to get {P_perf_t(ξ), F_perf_t(ξ)}.  Compute Φ_perf(ξ)
          using the same formula — this tracks the paper's (IV.12), since
          Φ_perf inherently depends on ξ via its own commitments.
3. Plot 2×2 grid of cost histograms + mean lines, matching Fig. 8's layout.
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import matplotlib.pyplot as plt

from data import load_dataset
# Reuse already-fitted generators (fitting is cached at module import)
from stoch_mpc import (
    lw_load, lw_price, ln_pif, gn_alpha,
    solve_stochastic_step,
    E_bar, P_bar, P_und, rho, dP_bar,
)
import stoch_mpc as sm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Experimental configuration
# ─────────────────────────────────────────────────────────────────────────────
MONTH_HOURS     = 720                         # M in paper
N_HORIZON       = 168                         # 7-day horizon (Fig 8 setting)
N_SCEN_MPC      = 20                          # |Ξ̄|  (paper uses 50)
N_REALIZATIONS  = 30                          # |Ξ|  (paper uses 1000)
GAMMA           = 1.0                         # per-stage discount (user-added)
SIGMA_CASES     = {                           # paper's two variants
    "undiscounted": 1.0,
    "discounted":   MONTH_HOURS / N_HORIZON,  # = 720/168 ≈ 4.286
}
SEED_STOCH      = 2024
SEED_REAL_BASE  = 10_000                      # per-realization seed offset

# Make sure the stochastic-MPC module uses the same month length
sm.sim_time = MONTH_HOURS
sm.N        = N_HORIZON

# Real NYISO realization (used as the "truth" when running stoch MPC once)
_ds    = load_dataset(sim_time=MONTH_HOURS)
pi_D   = _ds.pi_D
L_nyi  = _ds.L
pie_nyi = _ds.pi_e
pif_nyi = _ds.pi_f
a_nyi  = _ds.alpha


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parametrized perfect-information LP (single RH step)
# ─────────────────────────────────────────────────────────────────────────────
def solve_perfect_info_step(
    t: int,
    E0: float,
    D_prev: float,
    P_prev: float,
    L: np.ndarray,
    pie: np.ndarray,
    pif: np.ndarray,
    alpha: np.ndarray,
    sigma_t: float,
    gamma: float = 1.0,
    horizon_max: int = N_HORIZON,
    T_total: int = MONTH_HOURS,
):
    """
    RH LP at time t using the *true* future values of L, π^e, π^f, α that
    lie within [t, t+N-1].  Mirrors full_info_mpc.solve_one_step but adds
    explicit σ_t and γ parameters.

    Variable layout  x = [P_0..P_{h-1}, F_0..F_{h-1}, E_0..E_{h-1}, D]
      iP(k) = k              iF(k) = h+k              iE(k) = 2h+k            iD = 3h
    """
    h = min(horizon_max, T_total - t)
    n = 3 * h + 1
    iP = lambda k: k
    iF = lambda k: h + k
    iE = lambda k: 2 * h + k
    iD = 3 * h

    # Objective — paper eq. III.1 rearranged to a min:
    #   c[P_k] = -γ^k · π^e_k        c[F_k] = γ^k · (π^e_k α_k - π^f_k)   c[D] = π^D/σ_t
    c = np.zeros(n)
    gpow = gamma ** np.arange(h)
    for k in range(h):
        ae, af, ak = pie[t + k], pif[t + k], alpha[t + k]
        c[iP(k)] = -gpow[k] * ae
        c[iF(k)] =  gpow[k] * (ae * ak - af)
    c[iD] = pi_D / sigma_t

    bounds = (
        [(-P_und, P_bar)] * h +
        [(0.0,    P_bar)] * h +
        [(0.0,    E_bar)] * h +
        [(D_prev, None)]
    )

    # Equalities:  E_k - E_{k-1} + P_k - α_k F_k = (E0 if k=0 else 0)
    A_eq = np.zeros((h, n))
    b_eq = np.zeros(h)
    for k in range(h):
        ak = alpha[t + k]
        A_eq[k, iE(k)] =  1.0
        A_eq[k, iP(k)] =  1.0
        A_eq[k, iF(k)] = -ak
        if k == 0:
            b_eq[k] = E0
        else:
            A_eq[k, iE(k - 1)] = -1.0

    rows_A, rows_b = [], []
    for k in range(h):
        ak, Lk = alpha[t + k], L[t + k]

        # (1a) P + F ≤ P_bar         (1b) -P + F ≤ P_und
        row = np.zeros(n); row[iP(k)] = 1.0;  row[iF(k)] = 1.0; rows_A.append(row); rows_b.append(P_bar)
        row = np.zeros(n); row[iP(k)] = -1.0; row[iF(k)] = 1.0; rows_A.append(row); rows_b.append(P_und)

        # (4b) ρF + E_k ≤ E_bar ;  ρF - E_k ≤ 0
        row = np.zeros(n); row[iE(k)] = 1.0;  row[iF(k)] = rho; rows_A.append(row); rows_b.append(E_bar)
        row = np.zeros(n); row[iE(k)] = -1.0; row[iF(k)] = rho; rows_A.append(row); rows_b.append(0.0)

        # (4a) ρF + E_{k-1} ≤ E_bar ;  ρF - E_{k-1} ≤ 0    (use E0 if k=0)
        if k == 0:
            row = np.zeros(n); row[iF(k)] = rho; rows_A.append(row); rows_b.append(E_bar - E0)
            row = np.zeros(n); row[iF(k)] = rho; rows_A.append(row); rows_b.append(E0)
        else:
            row = np.zeros(n); row[iE(k-1)] = 1.0;  row[iF(k)] = rho; rows_A.append(row); rows_b.append(E_bar)
            row = np.zeros(n); row[iE(k-1)] = -1.0; row[iF(k)] = rho; rows_A.append(row); rows_b.append(0.0)

        # (5) Ramp:  |P_k - P_{k-1}| ≤ ΔP    (anchor at P_prev for k=0)
        if k == 0:
            row = np.zeros(n); row[iP(k)] =  1.0; rows_A.append(row); rows_b.append(dP_bar + P_prev)
            row = np.zeros(n); row[iP(k)] = -1.0; rows_A.append(row); rows_b.append(dP_bar - P_prev)
        else:
            row = np.zeros(n); row[iP(k)] =  1.0; row[iP(k-1)] = -1.0; rows_A.append(row); rows_b.append(dP_bar)
            row = np.zeros(n); row[iP(k)] = -1.0; row[iP(k-1)] =  1.0; rows_A.append(row); rows_b.append(dP_bar)

        # (7) No sell-back:  P + F ≤ L_k
        row = np.zeros(n); row[iP(k)] = 1.0; row[iF(k)] = 1.0; rows_A.append(row); rows_b.append(Lk)

        # Epigraph:  D ≥ L_k - P_k + α_k F_k  =>  -D - P + α F ≤ -L_k
        row = np.zeros(n); row[iD] = -1.0; row[iP(k)] = -1.0; row[iF(k)] = ak
        rows_A.append(row); rows_b.append(-Lk)

    A_ub = np.array(rows_A); b_ub = np.array(rows_b)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        return 0.0, 0.0, E0, D_prev, res.status
    x = res.x
    return x[iP(0)], x[iF(0)], x[iE(0)], x[iD], res.status


# ─────────────────────────────────────────────────────────────────────────────
# 3. Run one full RH MPC over the month
# ─────────────────────────────────────────────────────────────────────────────
def run_perfect_info_once(L, pie, pif, alpha, sigma_t, gamma=GAMMA, E0=None):
    """Perfect-info RH-MPC: returns trajectories P, F, d, E."""
    if E0 is None:
        E0 = E_bar / 2.0
    T = MONTH_HOURS
    E = np.zeros(T + 1); E[0] = E0
    P = np.zeros(T); F = np.zeros(T); d = np.zeros(T)
    D_run = 0.0; P_prev = 0.0
    for t in range(T):
        P_t, F_t, _E_next, _D_t, status = solve_perfect_info_step(
            t, E[t], D_run, P_prev, L, pie, pif, alpha,
            sigma_t=sigma_t, gamma=gamma,
        )
        if status != 0:
            P_t = F_t = 0.0
        # Apply with the TRUE α_t (same α is used in-horizon and for state update)
        dt = L[t] - P_t + alpha[t] * F_t
        E[t + 1] = float(np.clip(E[t] - P_t + alpha[t] * F_t, 0.0, E_bar))
        P[t] = P_t; F[t] = F_t; d[t] = dt
        D_run = max(D_run, dt)
        P_prev = P_t
    return dict(P=P, F=F, d=d, E=E, D_final=D_run)


def run_stochastic_once(sigma_t, gamma=GAMMA, n_scen=N_SCEN_MPC,
                        seed0=SEED_STOCH, E0=None,
                        L=L_nyi, pie=pie_nyi, pif=pif_nyi, alpha=a_nyi):
    """Stochastic RH-MPC with real NYISO data as 'environment'."""
    if E0 is None:
        E0 = E_bar / 2.0
    T = MONTH_HOURS
    E = np.zeros(T + 1); E[0] = E0
    P = np.zeros(T); F = np.zeros(T); d = np.zeros(T)
    D_run = 0.0; P_prev = 0.0
    for t in range(T):
        P_t, F_t, _E1m, _obj, status = solve_stochastic_step(
            t, E[t], D_run, P_prev,
            n_scen=n_scen, gamma=gamma, sigma_t=sigma_t,
            seed=seed0 + t,
        )
        if status != 0:
            P_t = F_t = 0.0
        # Apply with real α
        dt = L[t] - P_t + alpha[t] * F_t
        E[t + 1] = float(np.clip(E[t] - P_t + alpha[t] * F_t, 0.0, E_bar))
        P[t] = P_t; F[t] = F_t; d[t] = dt
        D_run = max(D_run, dt)
        P_prev = P_t
    return dict(P=P, F=F, d=d, E=E, D_final=D_run)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Realization generator & cost evaluator
# ─────────────────────────────────────────────────────────────────────────────
def sample_realization(M: int, seed: int):
    """
    Draw a single joint realization of (L, π^e, π^f, α) over M hours from the
    fitted distributions.  Seasonally-adjusted LW for L and π^e (starting at
    t=0 = start of the year, same slice the MPC sees); iid samples for π^f, α.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1 << 31, size=4)
    # sample_horizon returns (M, n_scenarios); we take n=1 and squeeze
    L_xi   = lw_load .sample_horizon(0, M, 1, seed=int(seeds[0]))[:, 0]
    pie_xi = lw_price.sample_horizon(0, M, 1, seed=int(seeds[1]))[:, 0]
    pif_xi = ln_pif.sample(shape=M, seed=int(seeds[2]))
    a_xi   = gn_alpha.sample(shape=M, seed=int(seeds[3]))
    # Clip to plausible physical bounds
    L_xi   = np.clip(L_xi,   1.0,  None)
    pie_xi = np.clip(pie_xi, 1e-4, None)
    pif_xi = np.clip(pif_xi, 1e-5, None)
    a_xi   = np.clip(a_xi,   -1.0, 1.0)
    return L_xi, pie_xi, pif_xi, a_xi


def cost_on_realization(P, F, L, pie, pif, alpha):
    """Paper's IV.11 cost formula (applied to any commitment policy)."""
    d       = L - P + alpha * F
    energy  = float(np.sum(pie * (alpha * F - P)))   # >0 = paying for grid draw
    fr_rev  = float(np.sum(pif * F))
    demand  = float(pi_D * d.max())
    return energy - fr_rev + demand, dict(energy=energy, fr=fr_rev, demand=demand,
                                          peak=float(d.max()))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main experiment
# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = {}   # results[("stoch"|"perf", case_name)] = list of Φ values

    # --- Pre-sample the |Ξ| evaluation realizations once (same set for all methods/σ).
    print(f"\nSampling {N_REALIZATIONS} evaluation realizations over M={MONTH_HOURS}h ...")
    realizations = [sample_realization(MONTH_HOURS, seed=SEED_REAL_BASE + r)
                    for r in range(N_REALIZATIONS)]

    for case_name, sigma_t in SIGMA_CASES.items():
        print("\n" + "=" * 72)
        print(f" CASE: σ = {sigma_t:g}  ({case_name})")
        print("=" * 72)

        # --- Stochastic MPC — one commitment trajectory per case
        t0 = time.time()
        print(f"  running stochastic MPC (once over {MONTH_HOURS}h, |Ξ̄|={N_SCEN_MPC}) ...")
        stoch_traj = run_stochastic_once(sigma_t=sigma_t, gamma=GAMMA)
        print(f"    done in {time.time()-t0:.1f}s, peak={stoch_traj['D_final']:.0f} kW, "
              f"F mean={stoch_traj['F'].mean():.0f} kW")

        # --- Evaluate Φ_stoch on each realization (fixed commitment)
        stoch_costs = []
        for r, (Lx, pex, pfx, ax) in enumerate(realizations):
            phi, _ = cost_on_realization(stoch_traj["P"], stoch_traj["F"],
                                         Lx, pex, pfx, ax)
            stoch_costs.append(phi)
        stoch_costs = np.array(stoch_costs)

        # --- Perfect-info MPC: one run PER realization (different commitments per ξ)
        print(f"  running perfect-info MPC ({N_REALIZATIONS} realizations) ...")
        t0 = time.time()
        perf_costs = []
        for r, (Lx, pex, pfx, ax) in enumerate(realizations):
            pi_traj = run_perfect_info_once(Lx, pex, pfx, ax,
                                            sigma_t=sigma_t, gamma=GAMMA)
            phi, _ = cost_on_realization(pi_traj["P"], pi_traj["F"],
                                         Lx, pex, pfx, ax)
            perf_costs.append(phi)
            if (r + 1) % 10 == 0 or r == N_REALIZATIONS - 1:
                dt = time.time() - t0
                print(f"    [{r+1:3d}/{N_REALIZATIONS}]  cum {dt:.0f}s  "
                      f"avg {dt/(r+1):.1f}s/realization")
        perf_costs = np.array(perf_costs)

        results[("stoch", case_name)] = stoch_costs
        results[("perf",  case_name)] = perf_costs

        print(f"\n  summary  σ={case_name}:")
        print(f"    Φ_stoch  mean=${stoch_costs.mean():,.0f}  "
              f"std=${stoch_costs.std():,.0f}  "
              f"p90=${np.quantile(stoch_costs, 0.9):,.0f}")
        print(f"    Φ_perf   mean=${perf_costs.mean():,.0f}  "
              f"std=${perf_costs.std():,.0f}  "
              f"p90=${np.quantile(perf_costs, 0.9):,.0f}")

    # ── Plot 2×2 grid (rows = σ variant, cols = method) ──────────────────
    plot_path = os.path.join(root, "fig8_replica.png")
    plot_fig8(results, plot_path)
    print(f"\nPlot saved -> {plot_path}")

    # Save raw data for downstream analysis
    npz_path = os.path.join(root, "fig8_replica.npz")
    np.savez(npz_path,
             stoch_undisc=results[("stoch", "undiscounted")],
             stoch_disc  =results[("stoch", "discounted")],
             perf_undisc =results[("perf",  "undiscounted")],
             perf_disc   =results[("perf",  "discounted")],
             meta=dict(M=MONTH_HOURS, N=N_HORIZON, S=N_SCEN_MPC,
                       n_real=N_REALIZATIONS, pi_D=pi_D))
    print(f"Raw costs saved -> {npz_path}")


def plot_fig8(results, path):
    """2×2 grid  (rows: undiscounted / discounted) × (cols: stoch / perf)."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
    titles = {("stoch", "undiscounted"): "Stochastic  (σ=1)",
              ("perf",  "undiscounted"): "Perfect Information  (σ=1)",
              ("stoch", "discounted"):   "Stochastic  (σ=M/N)",
              ("perf",  "discounted"):   "Perfect Information  (σ=M/N)"}
    colors = {"stoch": "steelblue", "perf": "seagreen"}

    # Common x-axis per row so the two methods are directly comparable
    for row, case in enumerate(["undiscounted", "discounted"]):
        row_vals = np.concatenate([results[("stoch", case)], results[("perf", case)]])
        lo, hi = row_vals.min(), row_vals.max()
        pad = 0.05 * (hi - lo)
        xlim = (lo - pad, hi + pad)
        for col, method in enumerate(["stoch", "perf"]):
            ax = axs[row, col]
            vals = results[(method, case)]
            bins = np.linspace(*xlim, 25)
            ax.hist(vals, bins=bins, density=True, alpha=0.75,
                    color=colors[method], edgecolor="black", linewidth=0.5)
            ax.axvline(vals.mean(), color="black", lw=1.5, ls="--",
                       label=f"Mean ${vals.mean():,.0f}")
            ax.set_title(titles[(method, case)], fontsize=11)
            ax.set_xlim(*xlim)
            ax.set_xlabel("Monthly total cost ($)", fontsize=9)
            if col == 0:
                ax.set_ylabel("Density", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Fig. 8 replica — 7-day horizon MPC (N={N_HORIZON}, M={MONTH_HOURS}, "
        f"|Ξ̄|={N_SCEN_MPC}, |Ξ|={N_REALIZATIONS})\n"
        f"Top: undiscounted (σ=1). Bottom: discounted (σ=M/N≈{MONTH_HOURS/N_HORIZON:.2f}).",
        fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

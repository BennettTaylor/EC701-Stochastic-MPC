r"""
Two-stage stochastic MPC for battery + frequency regulation.

Implements the scheme in Kumar et al. Sec. III (paper 6044.pdf).  At each
hour t we solve a two-stage stochastic program over the horizon
N_t = {t+1, ..., t+N}:

  first stage   (here-and-now)     :  P_{t+1},  F_{t+1}
  second stage  (recourse per ξ)   :  P_k(ξ), F_k(ξ), E_k(ξ),  k ∈ N_t \ {t+1}
                                      and   D(ξ) = max_k d_k(ξ)

The non-anticipativity constraints (III.10a)-(III.10b) force P_{t+1} and
F_{t+1} to be identical across all scenarios ξ ∈ Ξ̄, so the first-stage
commitments are common across the draw.  Scenarios for L_k, π^e_k, π^f_k,
α_k come from the generators fitted in stochatic_mpc.py:
    L       SeasonalLedoitWolfEstimator      (diurnal baseline + LW residual)
    π^e     SeasonalLedoitWolfEstimator      (same treatment)
    π^f     EmpiricalLogNormalSampler        (iid log-normal)
    α       EmpiricalGaussianSampler         (iid Gaussian clipped to [-1,1])

Objective (min) — paper's eq. III.1 rearranged (drop constant π^e·L, flip sign):
    min  E[ Σ_k γ^{k-1} (-π^e_k P_k(ξ) + (π^e_k α_k(ξ) - π^f_k(ξ)) F_k(ξ))
            + (π^D / σ_t) D(ξ) ]

γ is a per-stage discount factor (user-added; γ=1 reproduces the paper).
σ_t is the paper's demand-charge discounting factor: σ=1 is undiscounted
(conservative), σ=M/N discounts the horizon's peak to its share of the
billing period (matches Kumar et al. Table I discounted case).

LP variable layout — flat vector x of length 3·S·N + S:
    iP(s,k) = s·N + k              for s=0..S-1, k=0..N-1
    iF(s,k) = S·N + s·N + k
    iE(s,k) = 2·S·N + s·N + k
    iD(s)   = 3·S·N + s
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data import load_dataset
from stochatic_mpc import (
    SeasonalLedoitWolfEstimator,
    EmpiricalLogNormalSampler,
    EmpiricalGaussianSampler,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. System parameters  (match full_info_mpc.py so comparisons are apples-to-apples)
# ─────────────────────────────────────────────────────────────────────────────
E_bar    = 500.0    # kWh    battery capacity
P_bar    = 1000.0   # kW     max discharge
P_und    = 1000.0   # kW     max charge
rho      = 0.1      # kWh/kW SoC buffer per unit FR capacity
dP_bar   = 500.0    # kW/h   ramp limit
N        = 24       # prediction horizon [h]
M        = 720      # billing-period length [h ≈ 1 month]
SIGMA_T  = M / N    # = 30   paper's discounted case.  Use 1.0 for undiscounted.

# Stochastic-MPC knobs
N_SCEN   = 20       # |Ξ̄|  (paper uses 50; 20 is faster for dev/demo)
GAMMA    = 1.0      # per-stage discount in objective; γ=1 => paper's formulation
sim_time = 336      # 2 weeks.  Set to None for full year (slow: ~hours).

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset + fitted scenario generators (fit ONCE on the full year)
# ─────────────────────────────────────────────────────────────────────────────
_ds_full = load_dataset()                           # full 8760h year
_ds_sim  = load_dataset(sim_time=sim_time)          # truncated for simulation

pi_D     = _ds_full.pi_D
L_real   = _ds_sim.L
pie_real = _ds_sim.pi_e
pif_real = _ds_sim.pi_f
a_real   = _ds_sim.alpha
T        = _ds_sim.meta["T"]

print("Fitting scenario generators on full year …")
lw_load  = SeasonalLedoitWolfEstimator(hours_per_sample=168, window_days=14).fit(_ds_full.L)
lw_price = SeasonalLedoitWolfEstimator(hours_per_sample=168, window_days=14).fit(_ds_full.pi_e)
ln_pif   = EmpiricalLogNormalSampler().fit(_ds_full.pi_f)
gn_alpha = EmpiricalGaussianSampler().fit(_ds_full.alpha)
print(f"  load   LW shrinkage = {lw_load.lw.shrinkage_intensity:.3f}")
print(f"  price  LW shrinkage = {lw_price.lw.shrinkage_intensity:.3f}")
print(f"  π^f    log-normal   μ_log={ln_pif.mu_log:+.3f}  σ_log={ln_pif.sigma_log:.3f}")
print(f"  α      Gaussian     μ={gn_alpha.mu:+.4f}  σ={gn_alpha.sigma:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scenario sampler (paired to a given t, horizon, S) — returns (horizon, S)
# ─────────────────────────────────────────────────────────────────────────────
def sample_scenarios(t: int, horizon: int, n_scen: int, seed: int):
    """
    Draw n_scen joint scenarios of (L, π^e, π^f, α) over the horizon starting
    at t.  L and π^e carry full seasonal + LW covariance structure; π^f and α
    are iid samples from their fitted marginals.

    Returns four arrays each of shape (horizon, n_scen).
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1 << 31, size=4)

    L_scen   = lw_load .sample_horizon(t, horizon, n_scen, seed=int(seeds[0]))
    pie_scen = lw_price.sample_horizon(t, horizon, n_scen, seed=int(seeds[1]))
    pif_scen = ln_pif  .sample(shape=(horizon, n_scen), seed=int(seeds[2]))
    a_scen   = gn_alpha.sample(shape=(horizon, n_scen), seed=int(seeds[3]))

    # Guard against implausible samples that would make the LP infeasible
    # (negative loads, negative prices).  These are rare but possible in the
    # Gaussian tails.
    L_scen   = np.clip(L_scen,   1.0,  None)
    pie_scen = np.clip(pie_scen, 1e-4, None)
    pif_scen = np.clip(pif_scen, 1e-5, None)
    a_scen   = np.clip(a_scen,   -1.0, 1.0)
    return L_scen, pie_scen, pif_scen, a_scen


# ─────────────────────────────────────────────────────────────────────────────
# 4. Two-stage stochastic LP  (one MPC step)
# ─────────────────────────────────────────────────────────────────────────────
def solve_stochastic_step(
    t: int,
    E0: float,
    D_prev: float,
    P_prev: float,
    n_scen: int = N_SCEN,
    gamma: float = GAMMA,
    sigma_t: float = SIGMA_T,
    seed: int = 0,
):
    """
    Solve the two-stage stochastic program at time t.

    Inputs
    ------
    E0      : current SoC [kWh]
    D_prev  : peak-demand carryover D̂_t [kW]  (lower bound on D(ξ))
    P_prev  : last applied net power [kW]    (ramp constraint anchor for k=0)
    n_scen  : number of scenarios |Ξ̄|
    gamma   : per-stage discount (γ=1 => paper)
    sigma_t : demand-charge discount (1 or M/N)

    Returns
    -------
    P1, F1, E1_scen_mean, obj, status
       — P1, F1 are the first-stage commitments (identical across scenarios)
       — E1_scen_mean is the scenario-averaged SoC at end of stage 0 (purely
         diagnostic; the *actual* SoC update uses the real α realization)
       — obj is the LP objective value (expected horizon cost in $)
       — status is scipy.linprog's status code (0 = optimal)
    """
    horizon = min(N, sim_time - t)
    S = n_scen

    # ── Sample scenarios ─────────────────────────────────────────────────
    L_sc, pie_sc, pif_sc, a_sc = sample_scenarios(t, horizon, S, seed=seed)
    # All shape (horizon, S)

    # ── Variable index helpers ───────────────────────────────────────────
    SN = S * horizon
    nvar = 3 * SN + S

    def iP(s, k): return s * horizon + k
    def iF(s, k): return SN + s * horizon + k
    def iE(s, k): return 2 * SN + s * horizon + k
    def iD(s):    return 3 * SN + s

    # ── Objective ────────────────────────────────────────────────────────
    # c[iP(s,k)] =  (γ^k / S) * (-π^e_{s,k})        (discharge reduces purchase)
    # c[iF(s,k)] =  (γ^k / S) * (π^e_{s,k} α_{s,k} - π^f_{s,k})
    # c[iD(s)]   =  (1/S) * π^D / σ_t
    c = np.zeros(nvar)
    gamma_pow = gamma ** np.arange(horizon)               # (horizon,)
    w = gamma_pow / S                                     # per-stage weight
    for s in range(S):
        for k in range(horizon):
            c[iP(s, k)] = -w[k] * pie_sc[k, s]
            c[iF(s, k)] =  w[k] * (pie_sc[k, s] * a_sc[k, s] - pif_sc[k, s])
        c[iD(s)] = (1.0 / S) * (pi_D / sigma_t)

    # ── Equality constraints: SoC dynamics + non-anticipativity ──────────
    # Build with COO triplets for speed.
    Arow, Acol, Aval = [], [], []
    beq = []
    row = 0

    # SoC dynamics per (s, k)
    for s in range(S):
        for k in range(horizon):
            # E_{s,k} + P_{s,k} - α·F_{s,k} - E_{s,k-1} = (E0 if k==0 else 0)
            Arow += [row, row, row]
            Acol += [iE(s, k), iP(s, k), iF(s, k)]
            Aval += [1.0,      1.0,     -a_sc[k, s]]
            if k == 0:
                beq.append(E0)
            else:
                Arow.append(row); Acol.append(iE(s, k - 1)); Aval.append(-1.0)
                beq.append(0.0)
            row += 1

    # Non-anticipativity: P_{s,0} - P_{0,0} = 0 and F_{s,0} - F_{0,0} = 0
    # (s = 1..S-1).  S=1 trivially satisfies so we skip.
    for s in range(1, S):
        Arow += [row, row]; Acol += [iP(s, 0), iP(0, 0)]; Aval += [1.0, -1.0]
        beq.append(0.0); row += 1
        Arow += [row, row]; Acol += [iF(s, 0), iF(0, 0)]; Aval += [1.0, -1.0]
        beq.append(0.0); row += 1

    A_eq = sp.coo_matrix((Aval, (Arow, Acol)), shape=(row, nvar)).tocsr()
    b_eq = np.array(beq)

    # ── Inequality constraints ──────────────────────────────────────────
    Urow, Ucol, Uval = [], [], []
    bub = []
    row = 0

    def add_row(entries, rhs):
        """entries: list of (col, val); rhs: scalar."""
        nonlocal row
        for col, val in entries:
            Urow.append(row); Ucol.append(col); Uval.append(val)
        bub.append(rhs)
        row += 1

    for s in range(S):
        for k in range(horizon):
            Lk = L_sc[k, s]
            ak = a_sc[k, s]

            # (1a) P + F ≤ P_bar
            add_row([(iP(s, k), 1.0), (iF(s, k), 1.0)], P_bar)
            # (1b) -P + F ≤ P_und
            add_row([(iP(s, k), -1.0), (iF(s, k), 1.0)], P_und)

            # (2b) SoC buffer on E_k:   E_k + ρF_k ≤ E_bar  and  -E_k + ρF_k ≤ 0
            add_row([(iE(s, k), 1.0),  (iF(s, k), rho)], E_bar)
            add_row([(iE(s, k), -1.0), (iF(s, k), rho)], 0.0)

            # (2a) SoC buffer on E_{k-1}
            if k == 0:
                # E_{-1} = E0 is known scalar -> move to RHS
                add_row([(iF(s, k), rho)], E_bar - E0)
                add_row([(iF(s, k), rho)], E0)
            else:
                add_row([(iE(s, k - 1), 1.0),  (iF(s, k), rho)], E_bar)
                add_row([(iE(s, k - 1), -1.0), (iF(s, k), rho)], 0.0)

            # (3) Ramp
            if k == 0:
                #  P_0 - P_prev ≤ dP_bar   =>   P_0 ≤ dP_bar + P_prev
                add_row([(iP(s, k),  1.0)],  dP_bar + P_prev)
                add_row([(iP(s, k), -1.0)],  dP_bar - P_prev)
            else:
                add_row([(iP(s, k),  1.0), (iP(s, k - 1), -1.0)], dP_bar)
                add_row([(iP(s, k), -1.0), (iP(s, k - 1),  1.0)], dP_bar)

            # (4) No sell-back:  P + F ≤ L_k(ξ)
            add_row([(iP(s, k), 1.0), (iF(s, k), 1.0)], Lk)

            # Epigraph of max_k d_k:  D_s ≥ L_k - P + α F
            #                         -D_s - P + α F  ≤ -L_k
            add_row([(iD(s), -1.0), (iP(s, k), -1.0), (iF(s, k), ak)], -Lk)

    A_ub = sp.coo_matrix((Uval, (Urow, Ucol)), shape=(row, nvar)).tocsr()
    b_ub = np.array(bub)

    # ── Variable bounds ─────────────────────────────────────────────────
    lb = np.empty(nvar); ub = np.empty(nvar)
    lb[0:SN]            = -P_und;        ub[0:SN]            = P_bar         # iP
    lb[SN:2*SN]         = 0.0;           ub[SN:2*SN]         = P_bar         # iF
    lb[2*SN:3*SN]       = 0.0;           ub[2*SN:3*SN]       = E_bar         # iE
    lb[3*SN:]           = D_prev;        ub[3*SN:]           = np.inf        # iD  (peak carryover)
    bounds = list(zip(lb, ub))

    # ── Solve ───────────────────────────────────────────────────────────
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        return 0.0, 0.0, E0, 0.0, res.status

    x = res.x
    P1 = x[iP(0, 0)]                                    # first-stage (common)
    F1 = x[iF(0, 0)]
    E1_mean = float(np.mean([x[iE(s, 0)] for s in range(S)]))

    # Sanity check: non-anticipativity must actually hold post-solve.
    if S > 1:
        p1_vals = np.array([x[iP(s, 0)] for s in range(S)])
        f1_vals = np.array([x[iF(s, 0)] for s in range(S)])
        na_err  = max(np.ptp(p1_vals), np.ptp(f1_vals))
        if na_err > 1e-4:
            print(f"  [warn t={t}] non-anticipativity violated: "
                  f"P1 ptp={np.ptp(p1_vals):.2e}, F1 ptp={np.ptp(f1_vals):.2e}")

    return P1, F1, E1_mean, float(res.fun), res.status


# ─────────────────────────────────────────────────────────────────────────────
# 5. Receding-horizon simulation
# ─────────────────────────────────────────────────────────────────────────────
def simulate(gamma: float = GAMMA, sigma_t: float = SIGMA_T, n_scen: int = N_SCEN,
             seed0: int = 2024, log_every: int = 24):
    E_sim  = np.zeros(sim_time + 1)
    P_sim  = np.zeros(sim_time)
    F_sim  = np.zeros(sim_time)
    d_sim  = np.zeros(sim_time)
    cost_e = np.zeros(sim_time)
    rev_fr = np.zeros(sim_time)

    E_sim[0]   = E_bar / 2.0
    D_running  = 0.0
    P_prev     = 0.0
    demand_charges: list[float] = []
    status_bad = 0

    print(f"\nStochastic MPC simulation  "
          f"(T={sim_time}h, N={N}, |Ξ̄|={n_scen}, γ={gamma}, σ={sigma_t:g})")
    t0 = time.time()
    for t in range(sim_time):
        # Reset peak at start of each new billing period
        if t > 0 and t % M == 0:
            demand_charges.append(pi_D * D_running)
            print(f"  [billing end t={t}] demand = ${pi_D * D_running:,.2f} "
                  f"(peak {D_running:.1f} kW)")
            D_running = 0.0

        P_t, F_t, E_next_hat, obj, status = solve_stochastic_step(
            t, E_sim[t], D_running, P_prev,
            n_scen=n_scen, gamma=gamma, sigma_t=sigma_t,
            seed=seed0 + t,
        )
        if status != 0:
            status_bad += 1
            P_t, F_t = 0.0, 0.0

        # Apply decisions with the REAL realized α_t, L_t, π^e_t, π^f_t
        alpha_t = a_real[t]
        Lt      = L_real[t]
        E_next  = E_sim[t] - P_t + alpha_t * F_t
        # Clip SoC to physical bounds to absorb α-realization mismatch from MPC
        E_next  = float(np.clip(E_next, 0.0, E_bar))

        dt = Lt - P_t + alpha_t * F_t
        P_sim[t]     = P_t
        F_sim[t]     = F_t
        E_sim[t + 1] = E_next
        d_sim[t]     = dt
        D_running    = max(D_running, dt)
        cost_e[t]    = pie_real[t] * (alpha_t * F_t - P_t)   # matches full_info_mpc sign
        rev_fr[t]    = pif_real[t] * F_t
        P_prev       = P_t

        if t % log_every == 0:
            print(f"  t={t:4d}  SoC={E_sim[t]:5.0f}  "
                  f"P={P_t:+7.1f}  F={F_t:6.1f}  peak={D_running:5.0f}  "
                  f"obj={obj:+,.1f}")

    demand_charges.append(pi_D * D_running)
    wall = time.time() - t0

    totals = dict(
        energy=float(cost_e.sum()),
        fr=float(rev_fr.sum()),
        demand=float(sum(demand_charges)),
    )
    totals["net"] = totals["energy"] - totals["fr"] + totals["demand"]
    print(f"\n  Simulation done in {wall:.1f}s  ({wall/max(sim_time,1):.2f}s per MPC step)")
    if status_bad:
        print(f"  WARNING: {status_bad} LP solves failed; used safe P=F=0.")
    return dict(
        E=E_sim, P=P_sim, F=F_sim, d=d_sim, cost_e=cost_e, rev_fr=rev_fr,
        demand_charges=demand_charges, D_final=D_running,
        totals=totals, wall=wall,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting  (same style as full_info_mpc.py)
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(out, path, title_suffix=""):
    hours = np.arange(sim_time)
    fig   = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')
    gs    = gridspec.GridSpec(4, 1, hspace=0.45)

    GRID, ACCENT, BLUE, GREEN, RED, PURPLE, TEXT = (
        '#2a2a40', '#f0c040', '#4fc3f7', '#81c784', '#ef9a9a', '#ce93d8', '#e0e0e0')

    def style_ax(ax, ylabel, title):
        ax.set_facecolor('#16162a')
        ax.tick_params(colors=TEXT)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
        ax.set_title(title, color=ACCENT, fontsize=10, pad=4)
        for sp_ in ax.spines.values():
            sp_.set_color(GRID); sp_.set_linewidth(0.6)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_xlim(0, sim_time - 1)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(hours, L_real[:sim_time], color=BLUE,  lw=1.2, label='Load $L_t$')
    ax1.plot(hours, out["d"],          color=GREEN, lw=1.2, ls='--', label='Grid draw $d_t$')
    ax1.axhline(out["D_final"], color=RED, lw=0.9, ls=':',
                label=f'Peak D={out["D_final"]:.0f} kW')
    style_ax(ax1, 'kW', 'Load vs. Grid Draw')
    ax1.legend(fontsize=8, facecolor='#16162a', labelcolor=TEXT, loc='upper right')

    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(hours, out["P"], 0, where=out["P"] > 0, color=RED,  alpha=0.6, label='Discharge')
    ax2.fill_between(hours, out["P"], 0, where=out["P"] < 0, color=BLUE, alpha=0.6, label='Charge')
    ax2.plot(hours, out["F"], color=PURPLE, lw=1.2, label='FR capacity $F_t$')
    style_ax(ax2, 'kW', 'Battery Power & FR Capacity')
    ax2.legend(fontsize=8, facecolor='#16162a', labelcolor=TEXT, loc='upper right')

    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(np.arange(sim_time + 1), out["E"], color=ACCENT, alpha=0.25)
    ax3.plot(np.arange(sim_time + 1), out["E"], color=ACCENT, lw=1.3, label='SoC $E_t$')
    ax3.axhline(E_bar,     color=RED,  lw=0.8, ls='--', label=f'Max {E_bar:.0f} kWh')
    ax3.axhline(E_bar / 2, color=GRID, lw=0.8, ls=':',  label='50%')
    style_ax(ax3, 'kWh', 'Battery State of Charge')
    ax3.set_xlim(0, sim_time)
    ax3.legend(fontsize=8, facecolor='#16162a', labelcolor=TEXT)

    ax4  = fig.add_subplot(gs[3])
    ax4b = ax4.twinx()
    ax4.plot(hours, pie_real[:sim_time] * 1000, color=GREEN,  lw=1.0, label='Elec. price [$/MWh]')
    ax4.plot(hours, pif_real[:sim_time] * 1000, color=PURPLE, lw=1.0, label='FR price [$/MW-h]')
    ax4b.fill_between(hours, a_real[:sim_time], 0, color=BLUE, alpha=0.25, label=r'$\alpha_t$')
    ax4b.set_ylabel(r'$\alpha$ [-]', color=BLUE, fontsize=9)
    ax4b.tick_params(colors=BLUE)
    for sp_ in ax4b.spines.values():
        sp_.set_color(GRID); sp_.set_linewidth(0.6)
    style_ax(ax4, '$ / MWh', 'Market Prices & ISO FR Signal')
    ax4.set_xlabel('Hour', color=TEXT, fontsize=9)
    lines1, labs1 = ax4.get_legend_handles_labels()
    lines2, labs2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labs1 + labs2,
               fontsize=8, facecolor='#16162a', labelcolor=TEXT, loc='upper right')

    for ax in [ax1, ax2, ax3, ax4]:
        for d in range(1, sim_time // 24 + 1):
            ax.axvline(d * 24, color=GRID, lw=0.6, ls='--')

    t = out["totals"]
    fig.suptitle(
        f'Stochastic MPC {title_suffix}  |  {sim_time}h  (N={N}, |Ξ̄|={N_SCEN})\n'
        f'Net cost: ${t["net"]:,.2f}   '
        f'(energy ${t["energy"]:,.2f} − FR ${t["fr"]:,.2f} + demand ${t["demand"]:,.2f})',
        color=TEXT, fontsize=11, y=0.99)

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  plot saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = simulate(gamma=GAMMA, sigma_t=SIGMA_T, n_scen=N_SCEN)

    print("\n" + "=" * 60)
    t = out["totals"]
    print(f"  Energy cost  :  ${t['energy']:>10,.2f}")
    print(f"  FR revenue   : -${t['fr']:>10,.2f}")
    print(f"  Demand charge:  ${t['demand']:>10,.2f}   "
          f"({len(out['demand_charges'])} billing window(s))")
    print(f"  NET COST     :  ${t['net']:>10,.2f}")
    print("=" * 60)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_results(out, os.path.join(root, f"stoch_mpc_N={N}_S={N_SCEN}.png"),
                 title_suffix=f"(γ={GAMMA}, σ={SIGMA_T:g})")

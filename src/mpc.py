"""
Note that Claude was used to assist with writing this code.

Full Information MPC for Battery Energy Storage with Frequency Regulation (FR)
===============================================================================
Parameters (from paper):
  E_bar  = 0.5 MWh    Battery capacity
  P_bar  = 1 MW       Max discharge rate
  P_und  = 1 MW       Max charge rate
  rho    = 0.1        kWh/kW – min SoC buffer per unit FR capacity
  dP_bar = 0.5 MW/h   Max ramping limit
  N      = 24         Rolling horizon (hours)
  M      = 720        Billing-period length (hours ~ 1 month)
  sigma  = M/N = 30   Demand-charge amortisation factor


Decision variables (per step in the N-horizon LP):
  P_k  : net battery discharge [kW]  (negative = charging)
  F_k  : FR capacity sold to ISO [kW]
  E_k  : battery SoC [kWh]
  D    : auxiliary peak-load variable (linearises the max)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import linprog

from data import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
E_bar    = 500.0   # kWh  (0.5 MWh capacity)
P_bar    = 1000.0  # kW   (1 MW max discharge)
P_und    = 1000.0  # kW   (1 MW max charge)
rho      = 0.1     # kWh/kW
dP_bar   = 500.0   # kW/h (0.5 MW/h ramp limit)
N        = 24      # rolling MPC horizon (hours)
M        = 720     # billing period (hours)
sigma    = M / N   # = 30  amortisation factor
sim_time = 720     # total simulation timesteps (720 = 1 month, 8760 = 1 year)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CSV CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_csv_filename(sim_time, N, sigma, M):
    """Build a unique filename that encodes the simulation parameters."""
    return f"./csv/mpc_results_T{sim_time}_N{N}_sigma{int(sigma)}_M{M}.csv"


def save_results_to_csv(filename, E_sim, P_sim, F_sim, d_sim, cost_e, rev_fr,
                         L_data, pie_data, pif_data, alpha_data):
    """
    Write per-timestep simulation results to a CSV file.

    Columns:
        t        : timestep index
        E_soc    : battery SoC at START of timestep [kWh]  (E_sim[t])
        P_batt   : battery net discharge [kW]  (positive = discharge)
        F_fr     : FR capacity committed [kW]
        d_grid   : grid draw [kW]
        cost_e   : incremental energy cost [$]
        rev_fr   : FR revenue [$]
        net_cost : cost_e - rev_fr [$]
        L        : building load [kW]
        pi_e     : electricity price [$/kWh]
        pi_f     : FR price [$/kW]
        alpha    : ISO FR dispatch signal [-]
    """
    sim_len = len(P_sim)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t", "E_soc_kWh", "P_batt_kW", "F_fr_kW", "d_grid_kW",
            "cost_e_usd", "rev_fr_usd", "net_cost_usd",
            "L_kW", "pi_e_per_kWh", "pi_f_per_kW", "alpha"
        ])
        for t in range(sim_len):
            writer.writerow([
                t,
                round(E_sim[t],   4),
                round(P_sim[t],   4),
                round(F_sim[t],   4),
                round(d_sim[t],   4),
                round(cost_e[t],  6),
                round(rev_fr[t],  6),
                round(cost_e[t] - rev_fr[t], 6),
                round(L_data[t],     4),
                round(pie_data[t],   6),
                round(pif_data[t],   6),
                round(alpha_data[t], 6),
            ])
    print(f"Results saved -> {filename}")


def load_results_from_csv(filename):
    """
    Load previously saved simulation results from a CSV file.

    Returns a dict of numpy arrays keyed by column name, plus
    derived aggregate scalars for reporting.
    """
    arrays = {
        "t":             [],
        "E_soc_kWh":     [],
        "P_batt_kW":     [],
        "F_fr_kW":       [],
        "d_grid_kW":     [],
        "cost_e_usd":    [],
        "rev_fr_usd":    [],
        "net_cost_usd":  [],
        "L_kW":          [],
        "pi_e_per_kWh":  [],
        "pi_f_per_kW":   [],
        "alpha":         [],
    }
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in arrays:
                arrays[key].append(float(row[key]))

    return {k: np.array(v) for k, v in arrays.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. LP SOLVER FOR ONE MPC STEP (deterministic MPC)
# ─────────────────────────────────────────────────────────────────────────────
"""
t      : current timestep
E0     : current battery state of charge
D_prev : peak demand seen so far this billing period
L      : load for next N timesteps, full array (sliced internally)
pie    : electricity price, full array
pif    : FR capacity price, full array
alpha  : ISO FR dispatch signal, full array
"""

def solve_one_step(t, E0, D_prev, L, pie, pif, alpha, pi_D, sim_time_len):
    """
    Solve the full-information MPC LP for horizon [t, t + horizon - 1].

    Variables flattened as:
        x = [ P_0..P_{h-1},  F_0..F_{h-1},  E_1..E_h,  D ]

    Index helpers:
        iP(k) = k               -> index of P_k
        iF(k) = horizon + k     -> index of F_k
        iE(k) = 2*horizon + k   -> index of E_k  (k=0 means E_1, first predicted SoC)
        iD    = 3*horizon       -> index of D (scalar auxiliary for peak demand)

    Constraints implemented:
        (1a) P_k + F_k <= P_bar                      discharge rate limit
        (1b) -P_k + F_k <= P_und                     charge rate limit
        (2a) rho*F_k <= E_{k-1} <= E_bar - rho*F_k  SoC buffer on E_{k-1}
        (2b) rho*F_k <= E_k    <= E_bar - rho*F_k    SoC buffer on E_k
        (3)  |P_k - P_{k-1}| <= dP_bar               ramp limit
        (4)  P_k + F_k <= L_k                        no sell-back to grid
        (5)  D >= D_prev                              peak carryover (lower bound)
        (6a) 0 <= E_k <= E_bar
        (6b) -P_und <= P_k <= P_bar
        (6c) 0 <= F_k <= P_bar
        epigraph: D >= d_k = L_k - P_k + alpha_k*F_k  for all k

    Returns: (P_t, F_t, E_next, D_t, status)
    """
    horizon = min(N, sim_time_len - t)
    n       = 3 * horizon + 1

    iP = lambda k: k
    iF = lambda k: horizon + k
    iE = lambda k: 2 * horizon + k
    iD = 3 * horizon

    # ── Objective ─────────────────────────────────────────────────────────────
    # min  sum_k [ pie_k * d_k  -  pif_k * F_k ]  +  (pi_D / sigma) * D
    # d_k = L_k - P_k + alpha_k * F_k
    # L_k is a known constant, dropped from objective (no effect on solution)
    c = np.zeros(n)
    for k in range(horizon):
        ae = pie[t + k]
        af = pif[t + k]
        ak = alpha[t + k]
        c[iP(k)] = -ae
        c[iF(k)] =  ae * ak - af
    c[iD] = pi_D / sigma

    # ── Variable bounds ───────────────────────────────────────────────────────
    bounds = (
        [(-P_und, P_bar)] * horizon +   # P_k  (6b)
        [(0.0,    P_bar)] * horizon +   # F_k  (6c)
        [(0.0,    E_bar)] * horizon +   # E_k  (6a)
        [(D_prev, None)]                # D    (5: peak carryover via lower bound)
    )

    # ── Equality constraints: SoC dynamics ───────────────────────────────────
    # E_k = E_{k-1} - P_k + alpha_k * F_k
    A_eq = np.zeros((horizon, n))
    b_eq = np.zeros(horizon)
    for k in range(horizon):
        ak = alpha[t + k]
        A_eq[k, iE(k)] =  1.0
        A_eq[k, iP(k)] =  1.0
        A_eq[k, iF(k)] = -ak
        if k == 0:
            b_eq[k] = E0
        else:
            A_eq[k, iE(k - 1)] = -1.0

    # ── Inequality constraints ────────────────────────────────────────────────
    rows_A, rows_b = [], []

    for k in range(horizon):
        ak = alpha[t + k]
        Lk = L[t + k]

        # (1a) P_k + F_k <= P_bar
        row = np.zeros(n)
        row[iP(k)] = 1.0; row[iF(k)] = 1.0
        rows_A.append(row); rows_b.append(P_bar)

        # (1b) -P_k + F_k <= P_und
        row = np.zeros(n)
        row[iP(k)] = -1.0; row[iF(k)] = 1.0
        rows_A.append(row); rows_b.append(P_und)

        # (2b) SoC buffer on E_k:
        #   upper: E_k + rho*F_k <= E_bar
        row = np.zeros(n)
        row[iE(k)] = 1.0; row[iF(k)] = rho
        rows_A.append(row); rows_b.append(E_bar)
        #   lower: -E_k + rho*F_k <= 0
        row = np.zeros(n)
        row[iE(k)] = -1.0; row[iF(k)] = rho
        rows_A.append(row); rows_b.append(0.0)

        # (2a) SoC buffer on E_{k-1}:
        if k == 0:
            row = np.zeros(n)
            row[iF(k)] = rho
            rows_A.append(row); rows_b.append(E_bar - E0)
            row = np.zeros(n)
            row[iF(k)] = rho
            rows_A.append(row); rows_b.append(E0)
        else:
            row = np.zeros(n)
            row[iE(k-1)] = 1.0; row[iF(k)] = rho
            rows_A.append(row); rows_b.append(E_bar)
            row = np.zeros(n)
            row[iE(k-1)] = -1.0; row[iF(k)] = rho
            rows_A.append(row); rows_b.append(0.0)

        # (3) Ramp limits |P_k - P_{k-1}| <= dP_bar
        if k > 0:
            row = np.zeros(n)
            row[iP(k)] = 1.0; row[iP(k-1)] = -1.0
            rows_A.append(row); rows_b.append(dP_bar)
            row = np.zeros(n)
            row[iP(k-1)] = 1.0; row[iP(k)] = -1.0
            rows_A.append(row); rows_b.append(dP_bar)

        # (4) No sell-back: P_k + F_k <= L_k
        row = np.zeros(n)
        row[iP(k)] = 1.0; row[iF(k)] = 1.0
        rows_A.append(row); rows_b.append(Lk)

        # Epigraph: D >= L_k - P_k + ak*F_k  =>  -D - P_k + ak*F_k <= -L_k
        row = np.zeros(n)
        row[iD]    = -1.0
        row[iP(k)] = -1.0
        row[iF(k)] =  ak
        rows_A.append(row); rows_b.append(-Lk)

    A_ub = np.array(rows_A)
    b_ub = np.array(rows_b)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if res.status != 0:
        return 0.0, 0.0, E0, D_prev, res.status

    x = res.x
    return x[iP(0)], x[iF(0)], x[iE(0)], x[iD], res.status


# ─────────────────────────────────────────────────────────────────────────────
# 4. MPC SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_deterministic_mpc(L_data, pie_data, pif_data, alpha_data, pi_D, sim_time):
    """
    Execute the rolling-horizon deterministic MPC over sim_time steps.

    At each step t:
      1. Solve the N-step LP (solve_one_step) to get optimal P_t, F_t, E_{t+1}.
      2. Apply only the first action (receding horizon).
      3. Track running peak demand D_running; reset at each billing period.

    Returns
    -------
    E_sim  : SoC trajectory, shape (sim_time + 1,)  [kWh]
    P_sim  : battery discharge schedule, shape (sim_time,)  [kW]
    F_sim  : FR capacity schedule, shape (sim_time,)  [kW]
    d_sim  : grid draw trajectory, shape (sim_time,)  [kW]
    cost_e : per-step incremental energy cost, shape (sim_time,)  [$]
    rev_fr : per-step FR revenue, shape (sim_time,)  [$]
    demand_charges : list of demand charges per billing period  [$]
    D_final : final running peak demand at end of last billing period  [kW]
    """
    E_sim  = np.zeros(sim_time + 1)
    P_sim  = np.zeros(sim_time)
    F_sim  = np.zeros(sim_time)
    d_sim  = np.zeros(sim_time)
    cost_e = np.zeros(sim_time)
    rev_fr = np.zeros(sim_time)

    E_sim[0]  = E_bar / 2.0
    D_running = 0.0
    demand_charges = []

    print("Running Full-Information MPC simulation ...")
    print(f"  N={N}h MPC horizon | sim_time={sim_time}h | M={M}h billing | sigma={sigma:.0f}")
    print(f"  Battery: {E_bar} kWh | ramp {dP_bar} kW/h | rho={rho} kWh/kW\n")

    for t in range(sim_time):
        # Reset peak demand at the start of each new billing period
        if t % M == 0 and t > 0:
            demand_charges.append(pi_D * D_running)
            print(f"  [Billing period end t={t}] "
                  f"Demand charge: ${pi_D * D_running:,.2f} (peak {D_running:.1f} kW)")
            D_running = 0.0

        P_t, F_t, E_next, D_t, status = solve_one_step(
            t, E_sim[t], D_running,
            L_data, pie_data, pif_data, alpha_data,
            pi_D, sim_time
        )
        if status != 0:
            print(f"  [t={t:3d}] LP status={status}, using safe defaults.")

        P_sim[t]     = P_t
        F_sim[t]     = F_t
        E_sim[t + 1] = E_next
        d_sim[t]     = L_data[t] + alpha_data[t] * F_t - P_t
        D_running    = max(D_running, d_sim[t])

        # Marginal energy cost: pi_e * (alpha_k*F_k - P_k)
        # (pi_e * L_k is a constant baseline excluded per paper convention)
        cost_e[t] = pie_data[t] * (alpha_data[t] * F_t - P_t)
        rev_fr[t] = pif_data[t] * F_t

        if t % 24 == 0:
            print(f"  Day {t//24+1:3d}: SoC={E_sim[t]:.1f} kWh | peak_d={D_running:.1f} kW")

    # Record the final (possibly incomplete) billing period
    demand_charges.append(pi_D * D_running)

    return E_sim, P_sim, F_sim, d_sim, cost_e, rev_fr, demand_charges, D_running


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# Shared colour palette
_PALETTE = dict(
    GRID   = '#2a2a40',
    ACCENT = '#f0c040',
    BLUE   = '#4fc3f7',
    GREEN  = '#81c784',
    RED    = '#ef9a9a',
    PURPLE = '#ce93d8',
    TEXT   = '#e0e0e0',
    BG_FIG = '#0f0f1a',
    BG_AX  = '#16162a',
)


def _style_ax(ax, ylabel, title, xlim):
    """Apply dark-theme styling to a single axes."""
    p = _PALETTE
    ax.set_facecolor(p['BG_AX'])
    ax.tick_params(colors=p['TEXT'])
    ax.set_ylabel(ylabel, color=p['TEXT'], fontsize=9)
    ax.set_title(title, color=p['ACCENT'], fontsize=10, pad=4)
    for sp in ax.spines.values():
        sp.set_color(p['GRID']); sp.set_linewidth(0.6)
    ax.grid(True, color=p['GRID'], lw=0.5)
    ax.set_xlim(0, xlim)


def plot_timeseries(L, pi_e, pi_f, alpha, title="Data Overview"):
    """Four-panel overview of the input time-series data."""
    t = np.arange(len(L))
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t, L);    axs[0].set_ylabel("Load (kW)");               axs[0].set_title(title)
    axs[1].plot(t, pi_e); axs[1].set_ylabel("Electricity Price ($/kWh)")
    axs[2].plot(t, pi_f); axs[2].set_ylabel("FR Price ($/kW)")
    axs[3].plot(t, alpha);axs[3].set_ylabel("FR Signal α");             axs[3].set_xlabel("Hour")
    plt.tight_layout()
    plt.show()


def plot_mpc_results(E_sim, P_sim, F_sim, d_sim,
                     L_data, pie_data, pif_data, alpha_data,
                     D_final, total_cost, total_energy_cost,
                     total_fr_revenue, total_demand_charge,
                     sim_time, out_path):
    """
    Four-panel MPC results figure (dark theme).

    Panel 1 : Load vs. grid draw, with peak demand line
    Panel 2 : Battery power (discharge/charge) and FR capacity
    Panel 3 : Battery state of charge
    Panel 4 : Market prices and ISO FR signal
    """
    p = _PALETTE
    hours = np.arange(sim_time)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(p['BG_FIG'])
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    # Panel 1 – Load vs. grid draw
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(hours, L_data[:sim_time], color=p['BLUE'],  lw=1.2, label='Load $L_t$')
    ax1.plot(hours, d_sim,             color=p['GREEN'], lw=1.2, label='Grid draw $d_t$', ls='--')
    ax1.axhline(D_final, color=p['RED'], lw=0.9, ls=':', label=f'Peak D={D_final:.0f} kW')
    _style_ax(ax1, 'kW', 'Load vs. Grid Draw', sim_time - 1)
    ax1.legend(fontsize=8, facecolor=p['BG_AX'], labelcolor=p['TEXT'], loc='upper right')

    # Panel 2 – Battery power & FR capacity
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(hours, P_sim, 0, where=P_sim > 0, color=p['RED'],   alpha=0.6, label='Discharge')
    ax2.fill_between(hours, P_sim, 0, where=P_sim < 0, color=p['BLUE'],  alpha=0.6, label='Charge')
    ax2.plot(hours, F_sim, color=p['PURPLE'], lw=1.2, label='FR capacity $F_t$')
    _style_ax(ax2, 'kW', 'Battery Power & FR Capacity', sim_time - 1)
    ax2.legend(fontsize=8, facecolor=p['BG_AX'], labelcolor=p['TEXT'], loc='upper right')

    # Panel 3 – State of charge
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(np.arange(sim_time + 1), E_sim, color=p['ACCENT'], alpha=0.25)
    ax3.plot(np.arange(sim_time + 1), E_sim, color=p['ACCENT'], lw=1.3, label='SoC $E_t$')
    ax3.axhline(E_bar,     color=p['RED'],  lw=0.8, ls='--', label=f'Max {E_bar:.0f} kWh')
    ax3.axhline(E_bar / 2, color=p['GRID'], lw=0.8, ls=':',  label='50%')
    _style_ax(ax3, 'kWh', 'Battery State of Charge', sim_time)
    ax3.legend(fontsize=8, facecolor=p['BG_AX'], labelcolor=p['TEXT'])

    # Panel 4 – Prices & ISO signal
    ax4  = fig.add_subplot(gs[3])
    ax4b = ax4.twinx()
    ax4.plot(hours, pie_data[:sim_time] * 1000, color=p['GREEN'],  lw=1.0, label='Elec. price [$/MWh]')
    ax4.plot(hours, pif_data[:sim_time],         color=p['PURPLE'], lw=1.0, label='FR price [$/kW]')
    ax4b.fill_between(hours, alpha_data[:sim_time], 0,
                      color=p['BLUE'], alpha=0.25, label=r'$\alpha_t$ ISO signal')
    ax4b.set_ylabel(r'$\alpha$ [-]', color=p['BLUE'], fontsize=9)
    ax4b.tick_params(colors=p['BLUE'])
    for sp in ax4b.spines.values():
        sp.set_color(p['GRID']); sp.set_linewidth(0.6)
    _style_ax(ax4, '$ / (kW or MWh)', 'Market Prices & ISO FR Signal', sim_time - 1)
    ax4.set_xlabel('Hour', color=p['TEXT'], fontsize=9)
    lines1, labs1 = ax4.get_legend_handles_labels()
    lines2, labs2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labs1 + labs2,
               fontsize=8, facecolor=p['BG_AX'], labelcolor=p['TEXT'], loc='upper right')

    # Day dividers on all panels
    for ax in [ax1, ax2, ax3, ax4]:
        for d in range(1, sim_time // 24 + 1):
            ax.axvline(d * 24, color=p['GRID'], lw=0.6, ls='--')

    fig.suptitle(
        f'Full-Information MPC  |  {sim_time}h Battery + FR Simulation  (N={N}h horizon)\n'
        f'Net cost: ${total_cost:,.2f}   '
        f'(energy ${total_energy_cost:,.2f} '
        f'- FR ${total_fr_revenue:,.2f} '
        f'+ demand ${total_demand_charge:,.2f})',
        color=p['TEXT'], fontsize=11, y=0.99
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved -> {out_path}")


def print_cost_summary(total_energy_cost, total_fr_revenue,
                       total_demand_charge, demand_charges):
    """Print a formatted cost breakdown to stdout."""
    total_cost = total_energy_cost - total_fr_revenue + total_demand_charge
    print("\n" + "=" * 52)
    print(f"  Energy cost  :  ${total_energy_cost:>10.2f}")
    print(f"  FR revenue   : -${total_fr_revenue:>10.2f}")
    print(f"  Demand charge:  ${total_demand_charge:>10.2f}  "
          f"({len(demand_charges)} billing periods)")
    print(f"  NET COST     :  ${total_cost:>10.2f}")
    print("=" * 52)
    return total_cost


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load market / load data ───────────────────────────────────────────────
    _ds      = load_dataset(sim_time=sim_time)
    L_data   = _ds.L
    pie_data = _ds.pi_e
    pif_data = _ds.pi_f
    alpha_data = _ds.alpha
    pi_D     = _ds.pi_D
    actual_T = _ds.meta["T"]   # clamp to actual data length

    print("\nData sources:")
    for k, v in _ds.meta.items():
        print(f"  {k:14s}: {v}")

    # ── Check for cached results ──────────────────────────────────────────────
    csv_file = make_csv_filename(actual_T, N, sigma, M)

    if os.path.exists(csv_file):
        print(f"\nCached results found: {csv_file}")
        print("Loading from CSV — skipping simulation.\n")
        data = load_results_from_csv(csv_file)

        # Reconstruct arrays from cached data
        E_sim  = np.append(data["E_soc_kWh"], data["E_soc_kWh"][-1])  # approximate final SoC
        P_sim  = data["P_batt_kW"]
        F_sim  = data["F_fr_kW"]
        d_sim  = data["d_grid_kW"]
        cost_e = data["cost_e_usd"]
        rev_fr = data["rev_fr_usd"]

        total_energy_cost  = cost_e.sum()
        total_fr_revenue   = rev_fr.sum()
        D_final            = d_sim.max()
        total_demand_charge = pi_D * D_final   # approximate; exact value stored in original run
        demand_charges     = [total_demand_charge]

    else:
        print(f"\nNo cached results found for: {csv_file}")
        print("Running MPC simulation ...\n")

        plot_timeseries(L_data, pie_data, pif_data, alpha_data,
                        title=f"NYISO 2023 + Campus Load (T={actual_T}h)")

        # ── Run simulation ────────────────────────────────────────────────────
        (E_sim, P_sim, F_sim, d_sim,
         cost_e, rev_fr,
         demand_charges, D_final) = run_deterministic_mpc(
            L_data, pie_data, pif_data, alpha_data, pi_D, actual_T
        )

        total_energy_cost   = cost_e.sum()
        total_fr_revenue    = rev_fr.sum()
        total_demand_charge = sum(demand_charges)

        # ── Save results ──────────────────────────────────────────────────────
        save_results_to_csv(
            csv_file,
            E_sim, P_sim, F_sim, d_sim, cost_e, rev_fr,
            L_data, pie_data, pif_data, alpha_data
        )

    # ── Cost summary ──────────────────────────────────────────────────────────
    total_demand_charge = sum(demand_charges)
    total_cost = print_cost_summary(
        total_energy_cost, total_fr_revenue, total_demand_charge, demand_charges
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_plot = f"./figures/deterministic_results_N={N}_sigma={int(sigma)}.png"
    plot_mpc_results(
        E_sim, P_sim, F_sim, d_sim,
        L_data, pie_data, pif_data, alpha_data,
        D_final,
        total_cost, total_energy_cost, total_fr_revenue, total_demand_charge,
        actual_T, out_plot
    )


if __name__ == "__main__":
    main()
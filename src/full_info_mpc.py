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
sim_time = 720    # total simulation timesteps, run for a month  720, 8760 for year
# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(sim_time, seed=42):
    """
    Produce sim_time hours of fake-but-plausible operational data.

    L_t   : building load [kW]  - double-hump daily profile, lower on weekends
    pi_e  : electricity price [$/kWh] - tracks load shape + time-of-use
    pi_f  : FR capacity price [$/kW]  - higher during business hours
    alpha : hourly ISO FR signal in [-1, 1] - Ornstein-Uhlenbeck process
    """
    rng = np.random.default_rng(seed)
    h   = np.arange(sim_time) % 24
    dow = (np.arange(sim_time) // 24) % 7  # day of week

    morning = np.exp(-((h -  9) ** 2) / 8.0)
    evening = np.exp(-((h - 18) ** 2) / 6.0)
    weekday = np.where(dow < 5, 1.0, 0.65)

    # University campus scale: 18,000-34,000 kW (matching paper's real data)
    # Base ~22,000 kW + daily swings of ~8,000 kW peak-to-peak
    L = 22000 + 8000 * (0.4 * morning + 0.6 * evening)
    L = L * weekday + rng.normal(0, 500, sim_time)
    L = np.clip(L, 15000, 36000)

    tou  = 0.04 + 0.12 * (0.5 * morning + 0.8 * evening)
    pi_e = tou * weekday + rng.normal(0, 0.008, sim_time)
    pi_e = np.clip(pi_e, 0.01, 0.28)

    # PJM FR prices: paper gets ~$14k/month FR revenue at ~720kW avg commitment
    # => avg pi_f ~ $14,000 / (720kW * 720hrs) ~ $0.027/kW/hr
    biz  = np.where((h >= 8) & (h <= 20), 1.0, 0.35)
    pi_f = (0.015 + 0.030 * biz) * weekday + rng.normal(0, 0.003, sim_time)
    pi_f = np.clip(pi_f, 0.005, 0.08)

    alpha = np.zeros(sim_time)
    theta, sigma_ou = 0.25, 0.22
    for t in range(1, sim_time):
        alpha[t] = alpha[t-1] + theta * (0 - alpha[t-1]) + sigma_ou * rng.normal()
    alpha = np.clip(alpha, -1.0, 1.0)

    return L, pi_e, pi_f, alpha


def plot_timeseries(L, pi_e, pi_f, alpha, title="Synthetic Data"):
    t = np.arange(len(L))
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t, L);        axs[0].set_ylabel("Load (kW)");              axs[0].set_title(title)
    axs[1].plot(t, pi_e);     axs[1].set_ylabel("Electricity Price ($/kWh)")
    axs[2].plot(t, pi_f);     axs[2].set_ylabel("FR Price ($/kW)")
    axs[3].plot(t, alpha);    axs[3].set_ylabel("FR Signal α");            axs[3].set_xlabel("Hour")
    plt.tight_layout()
    plt.show()



# ─────────────────────────────────────────────────────────────────────────────
# 3. LP SOLVER FOR ONE MPC STEP, deterministic MPC 
# ─────────────────────────────────────────────────────────────────────────────
"""
t : current timestep 
E0 : current battery state of charge 
D_prev — peak demand seen so far this month
L — load for next N timesteps, array [t : t+horizon]
pie — electricity price, array [t : t+horizon]
pif — FR capacity price, array [t : t+horizon]
alpha — the full ISO FR dispatch signal array, sliced similarly
"""
def solve_one_step(t, E0, D_prev, L, pie, pif, alpha):
    """
    Solve the full-information MPC LP for horizon [t, t + horizon - 1].

    Variables flattened as:
        x = [ P_0..P_{h-1},  F_0..F_{h-1},  E_1..E_h,  D ]

    Index helpers:
        iP(k) = k               -> index of P_k
        iF(k) = horizon + k     -> index of F_k
        iE(k) = 2*horizon + k   -> index of E_k  (k=0 means E_1, the first predicted SoC)
        iD    = 3*horizon       -> index of D (scalar auxiliary for peak demand)

    Constraints implemented:
        (1a) P_k + F_k <= P_bar                    discharge rate limit
        (1b) -P_k + F_k <= P_und                   charge rate limit
        (2a) rho*F_k <= E_{k-1} <= E_bar - rho*F_k SoC buffer on E_{k-1}
        (2b) rho*F_k <= E_k    <= E_bar - rho*F_k  SoC buffer on E_k
        (3)  |P_k - P_{k-1}| <= dP_bar             ramp limit
        (4)  P_k + F_k <= L_k                      no sell-back to grid
        (5)  D >= D_prev                            peak carryover (via lower bound)
        (6a) 0 <= E_k <= E_bar
        (6b) -P_und <= P_k <= P_bar
        (6c) 0 <= F_k <= P_bar
        epigraph: D >= d_k = L_k - P_k + alpha_k*F_k  for all k  (linearises max_k d_k)

    Returns: (P_t, F_t, E_next, D_t, status)
    """
    # Horizon shrinks at end of simulation so we don't index out of bounds
    horizon = min(N, sim_time - t)
    n       = 3 * horizon + 1

    # Index helpers: map variable name + timestep -> position in flat x vector
    iP = lambda k: k
    iF = lambda k: horizon + k
    iE = lambda k: 2 * horizon + k
    iD = 3 * horizon

    # ── Objective ─────────────────────────────────────────────────────────────
    # min  sum_k [ pie_k * d_k  -  pif_k * F_k ]  +  (pi_D / sigma) * D
    # d_k = L_k - P_k + alpha_k * F_k
    # L_k is a known constant so drop from objective (doesn't affect solution)
    # => c[P_k] = -pie_k        (discharging P>0 reduces grid draw = saves cost)
    #    c[F_k] = pie_k*ak - pif_k
    #    c[D]   = pi_D / sigma
    c = np.zeros(n)
    for k in range(horizon):
        ae = pie[t + k]
        af = pif[t + k]
        ak = alpha[t + k]
        c[iP(k)] = -ae
        c[iF(k)] =  ae * ak - af
    c[iD] = pi_D / sigma

    # ── Variable bounds (6a, 6b, 6c) + D >= D_prev via lower bound ───────────
    bounds = (
        [(-P_und, P_bar)] * horizon +   # P_k  (6b)
        [(0.0,    P_bar)] * horizon +   # F_k  (6c)
        [(0.0,    E_bar)] * horizon +   # E_k  (6a)
        [(D_prev, None)]                # D    (5: peak carryover via lower bound)
    )

    # ── Equality constraints: SoC dynamics ───────────────────────────────────
    # E_k = E_{k-1} - P_k + alpha_k * F_k
    # Rearranged for linprog (Ax = b):
    #   E_k + P_k - alpha_k * F_k = E_{k-1}
    A_eq = np.zeros((horizon, n))
    b_eq = np.zeros(horizon)
    for k in range(horizon):
        ak = alpha[t + k]
        A_eq[k, iE(k)] =  1.0    # E_k
        A_eq[k, iP(k)] =  1.0    # P_k
        A_eq[k, iF(k)] = -ak     # -alpha_k * F_k
        if k == 0:
            b_eq[k] = E0          # E_{-1} = E0 (current SoC)
        else:
            A_eq[k, iE(k - 1)] = -1.0   # -E_{k-1}

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
            # E_{k-1} = E0 is a known scalar, so encode as bound on F_0 only
            #   upper: E0 + rho*F_0 <= E_bar  =>  rho*F_0 <= E_bar - E0
            row = np.zeros(n)
            row[iF(k)] = rho
            rows_A.append(row); rows_b.append(E_bar - E0)
            #   lower: -E0 + rho*F_0 <= 0  =>  rho*F_0 <= E0
            row = np.zeros(n)
            row[iF(k)] = rho
            rows_A.append(row); rows_b.append(E0)
        else:
            #   upper: E_{k-1} + rho*F_k <= E_bar
            row = np.zeros(n)
            row[iE(k-1)] = 1.0; row[iF(k)] = rho
            rows_A.append(row); rows_b.append(E_bar)
            #   lower: -E_{k-1} + rho*F_k <= 0
            row = np.zeros(n)
            row[iE(k-1)] = -1.0; row[iF(k)] = rho
            rows_A.append(row); rows_b.append(0.0)

        # (3) Ramp limits |P_k - P_{k-1}| <= dP_bar
        if k > 0:
            # up:   P_k - P_{k-1} <= dP_bar
            row = np.zeros(n)
            row[iP(k)] = 1.0; row[iP(k-1)] = -1.0
            rows_A.append(row); rows_b.append(dP_bar)
            # down: P_{k-1} - P_k <= dP_bar
            row = np.zeros(n)
            row[iP(k-1)] = 1.0; row[iP(k)] = -1.0
            rows_A.append(row); rows_b.append(dP_bar)

        # (4) No sell-back: P_k + F_k <= L_k
        row = np.zeros(n)
        row[iP(k)] = 1.0; row[iF(k)] = 1.0
        rows_A.append(row); rows_b.append(Lk)

        # Epigraph for max_k d_k:  D >= d_k = L_k - P_k + ak*F_k
        #   => -D - P_k + ak*F_k <= -L_k
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
# 4. SIMULATE
# ─────────────────────────────────────────────────────────────────────────────

# load data 
_ds = load_dataset(sim_time=sim_time)
L_data, pie_data, pif_data, alpha_data = _ds.L, _ds.pi_e, _ds.pi_f, _ds.alpha
pi_D   = _ds.pi_D        # $/kW/month, from data.py (ConEd tariff by default)
sim_time = _ds.meta["T"]  # clamp to actual data length
print("\nData sources:")
for k, v in _ds.meta.items():
    print(f"  {k:14s}: {v}")
print()
plot_timeseries(L_data, pie_data, pif_data, alpha_data,
                title=f"NYISO 2023 + Campus Load (T={sim_time}h)")

# All arrays sized to sim_time (the full simulation length)
E_sim  = np.zeros(sim_time + 1)
P_sim  = np.zeros(sim_time)
F_sim  = np.zeros(sim_time)
d_sim  = np.zeros(sim_time)
cost_e = np.zeros(sim_time)
rev_fr = np.zeros(sim_time)

E_sim[0]  = E_bar / 2.0
D_running = 0.0
demand_charges = []  # record each billing period's demand charge separately

print("Running Full-Information MPC simulation ...")
print(f"  N={N}h MPC horizon | sim_time={sim_time}h | M={M}h billing | sigma={sigma:.0f}")
print(f"  Battery: {E_bar} kWh | ramp {dP_bar} kW/h | rho={rho} kWh/kW\n")

for t in range(sim_time):
    # Reset peak demand at the start of each new billing period
    if t % M == 0 and t > 0:
        demand_charges.append(pi_D * D_running)
        print(f"  [Billing period end t={t}] Demand charge: ${pi_D * D_running:,.2f} (peak {D_running:.1f} kW)")
        D_running = 0.0

    P_t, F_t, E_next, D_t, status = solve_one_step(
        t, E_sim[t], D_running,
        L_data, pie_data, pif_data, alpha_data
    )
    if status != 0:
        print(f"  [t={t:3d}] LP status={status}, using safe defaults.")

    P_sim[t]     = P_t
    F_sim[t]     = F_t
    E_sim[t + 1] = E_next
    d_sim[t]     = L_data[t] + alpha_data[t] * F_t - P_t
    D_running    = max(D_running, d_sim[t])
    # Net energy cost = pi_e * (alpha_k*F_k - P_k)  [battery's marginal energy effect]
    # This matches the paper's convention: pi_e*L_k is a constant baseline excluded
    cost_e[t]    = pie_data[t] * (alpha_data[t] * F_t - P_t)
    rev_fr[t]    = pif_data[t] * F_t

    if t % 24 == 0:
        print(f"  Day {t//24+1:3d}: SoC={E_sim[t]:.1f} kWh | peak_d={D_running:.1f} kW")

# Add the final (incomplete) billing period's demand charge
demand_charges.append(pi_D * D_running)

total_energy_cost = cost_e.sum()
total_fr_revenue  = rev_fr.sum()
total_demand_charge = sum(demand_charges)
total_cost        = total_energy_cost - total_fr_revenue + total_demand_charge

print("\n" + "=" * 52)
print(f"  Energy cost  :  ${total_energy_cost:>10.2f}")
print(f"  FR revenue   : -${total_fr_revenue:>10.2f}")
print(f"  Demand charge:  ${total_demand_charge:>10.2f}  ({len(demand_charges)} billing periods)")
print(f"  NET COST     :  ${total_cost:>10.2f}")
print("=" * 52)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT
# ─────────────────────────────────────────────────────────────────────────────

hours = np.arange(sim_time)
fig   = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#0f0f1a')
gs    = gridspec.GridSpec(4, 1, hspace=0.45)

GRID   = '#2a2a40'
ACCENT = '#f0c040'
BLUE   = '#4fc3f7'
GREEN  = '#81c784'
RED    = '#ef9a9a'
PURPLE = '#ce93d8'
TEXT   = '#e0e0e0'


def style_ax(ax, ylabel, title):
    ax.set_facecolor('#16162a')
    ax.tick_params(colors=TEXT)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.set_title(title, color=ACCENT, fontsize=10, pad=4)
    for sp in ax.spines.values():
        sp.set_color(GRID); sp.set_linewidth(0.6)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_xlim(0, sim_time - 1)


ax1 = fig.add_subplot(gs[0])
ax1.plot(hours, L_data[:sim_time], color=BLUE,  lw=1.2, label='Load $L_t$')
ax1.plot(hours, d_sim,             color=GREEN, lw=1.2, label='Grid draw $d_t$', ls='--')
ax1.axhline(D_running, color=RED, lw=0.9, ls=':', label=f'Peak D={D_running:.0f} kW')
style_ax(ax1, 'kW', 'Load vs. Grid Draw')
ax1.legend(fontsize=8, facecolor='#16162a', labelcolor=TEXT, loc='upper right')

ax2 = fig.add_subplot(gs[1])
ax2.fill_between(hours, P_sim, 0, where=P_sim > 0, color=RED,   alpha=0.6, label='Discharge')
ax2.fill_between(hours, P_sim, 0, where=P_sim < 0, color=BLUE,  alpha=0.6, label='Charge')
ax2.plot(hours, F_sim, color=PURPLE, lw=1.2, label='FR capacity $F_t$')
style_ax(ax2, 'kW', 'Battery Power & FR Capacity')
ax2.legend(fontsize=8, facecolor='#16162a', labelcolor=TEXT, loc='upper right')

ax3 = fig.add_subplot(gs[2])
ax3.fill_between(np.arange(sim_time + 1), E_sim, color=ACCENT, alpha=0.25)
ax3.plot(np.arange(sim_time + 1), E_sim, color=ACCENT, lw=1.3, label='SoC $E_t$')
ax3.axhline(E_bar,     color=RED,  lw=0.8, ls='--', label=f'Max {E_bar:.0f} kWh')
ax3.axhline(E_bar / 2, color=GRID, lw=0.8, ls=':',  label='50%')
style_ax(ax3, 'kWh', 'Battery State of Charge')
ax3.set_xlim(0, sim_time)   # E_sim has sim_time+1 points
ax3.legend(fontsize=8, facecolor='#16162a', labelcolor=TEXT)

ax4  = fig.add_subplot(gs[3])
ax4b = ax4.twinx()
ax4.plot(hours, pie_data[:sim_time] * 1000, color=GREEN,  lw=1.0, label='Elec. price [$/MWh]')
ax4.plot(hours, pif_data[:sim_time],        color=PURPLE, lw=1.0, label='FR price [$/kW]')
ax4b.fill_between(hours, alpha_data[:sim_time], 0, color=BLUE, alpha=0.25, label=r'$\alpha_t$ ISO signal')
ax4b.set_ylabel(r'$\alpha$ [-]', color=BLUE, fontsize=9)
ax4b.tick_params(colors=BLUE)
for sp in ax4b.spines.values():
    sp.set_color(GRID); sp.set_linewidth(0.6)
style_ax(ax4, '$ / (kW or MWh)', 'Market Prices & ISO FR Signal')
ax4.set_xlabel('Hour', color=TEXT, fontsize=9)
lines1, labs1 = ax4.get_legend_handles_labels()
lines2, labs2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labs1 + labs2,
           fontsize=8, facecolor='#16162a', labelcolor=TEXT, loc='upper right')

# Day dividers
for ax in [ax1, ax2, ax3, ax4]:
    for d in range(1, sim_time // 24 + 1):
        ax.axvline(d * 24, color=GRID, lw=0.6, ls='--')

fig.suptitle(
    f'Full-Information MPC  |  {sim_time}h Battery + FR Simulation  (N={N}h horizon)\n'
    f'Net cost: ${total_cost:,.2f}   '
    f'(energy ${total_energy_cost:,.2f} '
    f'- FR ${total_fr_revenue:,.2f} '
    f'+ demand ${total_demand_charge:,.2f})',
    color=TEXT, fontsize=11, y=0.99
)

out_plot = './figures/deterministic_results_N='+str(N)+'_sigma='+str(sigma)+'.png'
plt.savefig(out_plot, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved -> {out_plot}")

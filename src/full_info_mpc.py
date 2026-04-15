"""
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

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
E_bar  = 500.0   # kWh  (0.5 MWh capacity)
P_bar  = 1000.0  # kW   (1 MW max discharge)
P_und  = 1000.0  # kW   (1 MW max charge)
rho    = 0.1     # kWh/kW
dP_bar = 500.0   # kW/h (0.5 MW/h ramp limit)
pi_D   = 15.0    # $/kW  demand charge rate
N      = 24      # rolling horizon (hours)
M      = 720     # billing period (hours)
sigma  = M / N   # = 30
T      = 168     # simulation horizon (1 week)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(T, seed=42):
    """
    Produce a week of fake-but-plausible operational data.

    L_t   : building load [kW]  - double-hump daily profile, lower on weekends
    pi_e  : electricity price [$/kWh] - tracks load shape + time-of-use
    pi_f  : FR capacity price [$/kW]  - higher during business hours
    alpha : hourly ISO FR signal in [-1, 1] - Ornstein-Uhlenbeck process
    """
    rng = np.random.default_rng(seed)
    h   = np.arange(T) % 24
    dow = (np.arange(T) // 24) % 7

    morning = np.exp(-((h -  9) ** 2) / 8.0)
    evening = np.exp(-((h - 18) ** 2) / 6.0)
    weekday = np.where(dow < 5, 1.0, 0.65)

    L = 200 + 350 * (0.5 * morning + 0.8 * evening)
    L = L * weekday + rng.normal(0, 15, T)
    L = np.clip(L, 50, 950)

    tou  = 0.04 + 0.12 * (0.5 * morning + 0.8 * evening)
    pi_e = tou * weekday + rng.normal(0, 0.008, T)
    pi_e = np.clip(pi_e, 0.01, 0.28)

    biz  = np.where((h >= 8) & (h <= 20), 1.0, 0.35)
    pi_f = (4.0 + 9.0 * biz) * weekday + rng.normal(0, 0.4, T)
    pi_f = np.clip(pi_f, 0.3, 18.0)

    alpha = np.zeros(T)
    theta, sigma_ou = 0.25, 0.22
    for t in range(1, T):
        alpha[t] = alpha[t-1] + theta * (0 - alpha[t-1]) + sigma_ou * rng.normal()
    alpha = np.clip(alpha, -1.0, 1.0)

    return L, pi_e, pi_f, alpha


L_data, pie_data, pif_data, alpha_data = generate_data(T)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LP SOLVER FOR ONE MPC STEP
# ─────────────────────────────────────────────────────────────────────────────

def solve_one_step(t, E0, D_prev, L, pie, pif, alpha):
    """
    Solve the full-information MPC LP for horizon [t, t + horizon - 1].

    Variables flattened as:
        x = [ P_0..P_{h-1},  F_0..F_{h-1},  E_1..E_h,  D ]

    Returns: (P_t, F_t, E_next, D_t, status)
    """
    horizon = min(N, T - t)
    n       = 3 * horizon + 1

    iP = lambda k: k
    iF = lambda k: horizon + k
    iE = lambda k: 2 * horizon + k
    iD = 3 * horizon

    # Objective
    c = np.zeros(n)
    for k in range(horizon):
        ae = pie[t + k]
        af = pif[t + k]
        ak = alpha[t + k]
        c[iP(k)] = -ae
        c[iF(k)] =  ae * ak - af
    c[iD] = sigma * pi_D

    # Bounds
    bounds = (
        [(-P_und, P_bar)] * horizon +
        [(0.0,    None)]  * horizon +
        [(0.0,    E_bar)] * horizon +
        [(D_prev, None)]
    )

    # Equality: SoC dynamics  E_{k+1} = E_k - P_k - alpha_k * F_k
    A_eq = np.zeros((horizon, n))
    b_eq = np.zeros(horizon)
    for k in range(horizon):
        ak = alpha[t + k]
        A_eq[k, iE(k)] =  1.0
        A_eq[k, iP(k)] =  1.0
        A_eq[k, iF(k)] =  ak
        if k == 0:
            b_eq[k] = E0
        else:
            A_eq[k, iE(k - 1)] = -1.0

    # Inequalities
    rows_A, rows_b = [], []
    for k in range(horizon):
        ak = alpha[t + k]
        Lk = L[t + k]

        # FR buffer upper: E_{k+1} + rho*F_k <= E_bar
        row = np.zeros(n)
        row[iE(k)] = 1.0; row[iF(k)] = rho
        rows_A.append(row); rows_b.append(E_bar)

        # FR buffer lower: -E_{k+1} + rho*F_k <= 0
        row = np.zeros(n)
        row[iE(k)] = -1.0; row[iF(k)] = rho
        rows_A.append(row); rows_b.append(0.0)

        # Ramp up
        if k > 0:
            row = np.zeros(n)
            row[iP(k)] = 1.0; row[iP(k-1)] = -1.0
            rows_A.append(row); rows_b.append(dP_bar)

        # Ramp down
        if k > 0:
            row = np.zeros(n)
            row[iP(k-1)] = 1.0; row[iP(k)] = -1.0
            rows_A.append(row); rows_b.append(dP_bar)

        # D >= d_k = Lk + ak*F_k - P_k
        row = np.zeros(n)
        row[iD]    = -1.0
        row[iF(k)] =  ak
        row[iP(k)] = -1.0
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

E_sim  = np.zeros(T + 1)
P_sim  = np.zeros(T)
F_sim  = np.zeros(T)
d_sim  = np.zeros(T)
cost_e = np.zeros(T)
rev_fr = np.zeros(T)

E_sim[0]  = E_bar / 2.0
D_running = 0.0

print("Running Full-Information MPC simulation ...")
print(f"  N={N}h horizon | M={M}h billing | sigma={sigma:.0f}")
print(f"  Battery: {E_bar} kWh | ramp {dP_bar} kW/h | rho={rho} kWh/kW\n")

for t in range(T):
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
    cost_e[t]    = pie_data[t] * d_sim[t]
    rev_fr[t]    = pif_data[t] * F_t

    if t % 24 == 0:
        print(f"  Day {t//24+1}: SoC={E_sim[t]:.1f} kWh | peak_d={D_running:.1f} kW")

total_energy_cost = cost_e.sum()
total_fr_revenue  = rev_fr.sum()
demand_charge     = pi_D * D_running
total_cost        = total_energy_cost - total_fr_revenue + demand_charge

print("\n" + "=" * 52)
print(f"  Energy cost  :  ${total_energy_cost:>10.2f}")
print(f"  FR revenue   : -${total_fr_revenue:>10.2f}")
print(f"  Demand charge:  ${demand_charge:>10.2f}  (peak {D_running:.1f} kW)")
print(f"  NET COST     :  ${total_cost:>10.2f}")
print("=" * 52)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT
# ─────────────────────────────────────────────────────────────────────────────

hours = np.arange(T)
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
    ax.set_xlim(0, T - 1)


ax1 = fig.add_subplot(gs[0])
ax1.plot(hours, L_data, color=BLUE,  lw=1.2, label='Load $L_t$')
ax1.plot(hours, d_sim,  color=GREEN, lw=1.2, label='Grid draw $d_t$', ls='--')
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
ax3.fill_between(np.arange(T + 1), E_sim, color=ACCENT, alpha=0.25)
ax3.plot(np.arange(T + 1), E_sim, color=ACCENT, lw=1.3, label='SoC $E_t$')
ax3.axhline(E_bar,     color=RED,  lw=0.8, ls='--', label=f'Max {E_bar:.0f} kWh')
ax3.axhline(E_bar / 2, color=GRID, lw=0.8, ls=':',  label='50%')
style_ax(ax3, 'kWh', 'Battery State of Charge')
ax3.legend(fontsize=8, facecolor='#16162a', labelcolor=TEXT)

ax4  = fig.add_subplot(gs[3])
ax4b = ax4.twinx()
ax4.plot(hours, pie_data * 1000, color=GREEN,  lw=1.0, label='Elec. price [$/MWh]')
ax4.plot(hours, pif_data,        color=PURPLE, lw=1.0, label='FR price [$/kW]')
ax4b.fill_between(hours, alpha_data, 0, color=BLUE, alpha=0.25, label=r'$\alpha_t$ ISO signal')
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

for ax in [ax1, ax2, ax3, ax4]:
    for d in range(1, 7):
        ax.axvline(d * 24, color=GRID, lw=0.6, ls='--')

fig.suptitle(
    f'Full-Information MPC  |  1-Week Battery + FR Simulation\n'
    f'Net cost: ${total_cost:,.2f}   '
    f'(energy ${total_energy_cost:,.2f} '
    f'- FR ${total_fr_revenue:,.2f} '
    f'+ demand ${demand_charge:,.2f})',
    color=TEXT, fontsize=11, y=0.99
)

out_plot = './mpc_results.png'
plt.savefig(out_plot, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved -> {out_plot}")
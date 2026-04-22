"""
Base class for receding-horizon MPC controllers + the single-path LP builder
shared by deterministic and perfect-information MPC.

Design
------
`MPCController` owns:
    - system parameters (battery capacity, ramp, etc.)
    - the RH simulate() loop: calls solve_step(), applies (P, F) to the true
      physics using real realized signals, accumulates cost / peak demand.
    - per-realization cost accounting.

Subclasses implement only `solve_step(t, E0, D_prev, P_prev) -> StepResult`.
They are free to use point forecasts (deterministic), scenario draws
(stochastic), or the true future (perfect info) — the base class doesn't
know or care.

The single-path LP builder `solve_single_path_lp` is a module-level helper
used by both DeterministicMPC and PerfectInfoMPC — same LP structure, just
different forecast data fed in.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.optimize import linprog


# ─────────────────────────────────────────────────────────────────────────────
# System parameters  (paper's Table of simulation constants)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SystemParams:
    E_bar:  float = 500.0    # kWh    battery capacity
    P_bar:  float = 1000.0   # kW     max discharge
    P_und:  float = 1000.0   # kW     max charge
    rho:    float = 0.1      # kWh/kW SoC buffer per unit FR capacity
    dP_bar: float = 500.0    # kW/h   ramp limit


DEFAULT_PARAMS = SystemParams()


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    P: float
    F: float
    status: int = 0
    info: dict = field(default_factory=dict)


@dataclass
class SimResult:
    method: str
    E: np.ndarray             # (T+1,)  SoC trajectory
    P: np.ndarray             # (T,)    net battery power [kW]
    F: np.ndarray             # (T,)    FR capacity [kW]
    d: np.ndarray             # (T,)    grid draw [kW]
    cost_e: np.ndarray        # (T,)    marginal energy cost per hour [$]
    rev_fr: np.ndarray        # (T,)    FR revenue per hour [$]
    demand_charges: list      # one $ entry per billing period
    D_final: float            # final running peak in the last billing period
    totals: dict              # {energy, fr, demand, net}
    wall: float               # wall-clock seconds
    # Real realizations the controller was run against (for CSV export)
    L: np.ndarray
    pi_e: np.ndarray
    pi_f: np.ndarray
    alpha: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────
class MPCController(ABC):
    """Receding-horizon MPC base class."""

    name: str = "abstract"

    def __init__(
        self,
        horizon: int,
        T: int,
        pi_D: float,
        sigma: float = 1.0,
        gamma: float = 1.0,
        params: SystemParams = DEFAULT_PARAMS,
        billing_hours: int = 720,
        E0: float | None = None,
    ):
        self.N = horizon
        self.T = T
        self.pi_D = pi_D
        self.sigma = sigma
        self.gamma = gamma
        self.p = params
        self.M = billing_hours
        self.E0 = E0 if E0 is not None else params.E_bar / 2.0

    # Subclass must implement this
    @abstractmethod
    def solve_step(
        self, t: int, E0: float, D_prev: float, P_prev: float
    ) -> StepResult:
        """Return the first-stage (P_{t+1}, F_{t+1}) commitment."""

    def simulate(
        self,
        L: np.ndarray,
        pi_e: np.ndarray,
        pi_f: np.ndarray,
        alpha: np.ndarray,
        verbose: bool = False,
    ) -> SimResult:
        """
        Receding-horizon simulation over T hours against REAL realizations.

        The controller's internal forecasts (means, scenarios, or truth) are
        its own business; state propagation is always against the real
        (L, pi_e, pi_f, alpha) arrays passed here.
        """
        T = self.T
        E = np.zeros(T + 1); E[0] = self.E0
        P = np.zeros(T); F = np.zeros(T); d = np.zeros(T)
        cost_e = np.zeros(T); rev_fr = np.zeros(T)
        demand_charges: list[float] = []

        D_run = 0.0
        P_prev = 0.0
        n_status_bad = 0

        t0 = time.time()
        for t in range(T):
            # Reset peak at the start of each new billing period
            if t > 0 and t % self.M == 0:
                demand_charges.append(self.pi_D * D_run)
                if verbose:
                    print(f"    [{self.name}] billing end t={t}  "
                          f"demand=${self.pi_D * D_run:,.0f}  peak={D_run:.0f} kW")
                D_run = 0.0

            step = self.solve_step(t, E[t], D_run, P_prev)
            if step.status != 0:
                n_status_bad += 1
                P_t, F_t = 0.0, 0.0
            else:
                P_t, F_t = step.P, step.F

            # Apply with the TRUE realized α, L, π^e, π^f
            a_t = alpha[t]
            L_t = L[t]
            E[t + 1] = float(np.clip(E[t] - P_t + a_t * F_t, 0.0, self.p.E_bar))
            d_t = L_t - P_t + a_t * F_t

            P[t] = P_t; F[t] = F_t; d[t] = d_t
            cost_e[t] = pi_e[t] * (a_t * F_t - P_t)
            rev_fr[t] = pi_f[t] * F_t
            D_run = max(D_run, d_t)
            P_prev = P_t

            if verbose and t % 24 == 0:
                print(f"    [{self.name}] t={t:4d}  SoC={E[t]:6.0f}  "
                      f"P={P_t:+7.1f}  F={F_t:6.1f}  peak={D_run:7.0f}")

        demand_charges.append(self.pi_D * D_run)
        wall = time.time() - t0

        totals = dict(
            energy = float(cost_e.sum()),
            fr     = float(rev_fr.sum()),
            demand = float(sum(demand_charges)),
        )
        totals["net"] = totals["energy"] - totals["fr"] + totals["demand"]

        if verbose and n_status_bad:
            print(f"    [{self.name}] WARNING: {n_status_bad} LP solves failed")

        return SimResult(
            method=self.name,
            E=E, P=P, F=F, d=d, cost_e=cost_e, rev_fr=rev_fr,
            demand_charges=demand_charges, D_final=D_run,
            totals=totals, wall=wall,
            L=L, pi_e=pi_e, pi_f=pi_f, alpha=alpha,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Single-path LP builder — used by deterministic & perfect-info MPC
# ─────────────────────────────────────────────────────────────────────────────
def solve_single_path_lp(
    horizon: int,
    E0: float,
    D_prev: float,
    P_prev: float,
    L: np.ndarray,
    pi_e: np.ndarray,
    pi_f: np.ndarray,
    alpha: np.ndarray,
    params: SystemParams,
    pi_D: float,
    sigma: float,
    gamma: float = 1.0,
) -> Tuple[float, float, int]:
    """
    Solve the deterministic / perfect-info horizon LP.

    The only difference between deterministic and perfect-info MPC is what
    goes into (L, pi_e, pi_f, alpha) — mean forecasts vs. true future.  The
    LP itself is identical.

    Variable layout  x = [P_0..P_{h-1}, F_0..F_{h-1}, E_0..E_{h-1}, D]
        iP(k) = k              iF(k) = h+k              iE(k) = 2h+k          iD = 3h

    Returns (P_0, F_0, status).  status == 0 means optimal.
    """
    h = horizon
    p = params
    n = 3 * h + 1

    def iP(k): return k
    def iF(k): return h + k
    def iE(k): return 2 * h + k
    iD = 3 * h

    # ── Objective ───────────────────────────────────────────────────────────
    # c[P_k] = -γ^k · π^e_k
    # c[F_k] =  γ^k · (π^e_k α_k - π^f_k)
    # c[D]   =  π^D / σ
    c = np.zeros(n)
    gpow = gamma ** np.arange(h)
    for k in range(h):
        c[iP(k)] = -gpow[k] * pi_e[k]
        c[iF(k)] =  gpow[k] * (pi_e[k] * alpha[k] - pi_f[k])
    c[iD] = pi_D / sigma

    # ── Variable bounds ─────────────────────────────────────────────────────
    bounds = (
        [(-p.P_und, p.P_bar)] * h +
        [(0.0,      p.P_bar)] * h +
        [(0.0,      p.E_bar)] * h +
        [(D_prev,   None)]
    )

    # ── Equality constraints: SoC dynamics ──────────────────────────────────
    # E_k - E_{k-1} + P_k - α_k F_k = (E0 if k==0 else 0)
    A_eq = np.zeros((h, n))
    b_eq = np.zeros(h)
    for k in range(h):
        A_eq[k, iE(k)] =  1.0
        A_eq[k, iP(k)] =  1.0
        A_eq[k, iF(k)] = -alpha[k]
        if k == 0:
            b_eq[k] = E0
        else:
            A_eq[k, iE(k - 1)] = -1.0

    # ── Inequality constraints ──────────────────────────────────────────────
    rows_A, rows_b = [], []
    for k in range(h):
        ak = alpha[k]; Lk = L[k]

        # (1a) P + F ≤ P_bar          (1b) -P + F ≤ P_und
        row = np.zeros(n); row[iP(k)] =  1.0; row[iF(k)] = 1.0
        rows_A.append(row); rows_b.append(p.P_bar)
        row = np.zeros(n); row[iP(k)] = -1.0; row[iF(k)] = 1.0
        rows_A.append(row); rows_b.append(p.P_und)

        # (2b) SoC buffer on E_k:  E_k + ρF ≤ E_bar ; -E_k + ρF ≤ 0
        row = np.zeros(n); row[iE(k)] =  1.0; row[iF(k)] = p.rho
        rows_A.append(row); rows_b.append(p.E_bar)
        row = np.zeros(n); row[iE(k)] = -1.0; row[iF(k)] = p.rho
        rows_A.append(row); rows_b.append(0.0)

        # (2a) SoC buffer on E_{k-1} (use E0 if k==0)
        if k == 0:
            row = np.zeros(n); row[iF(k)] = p.rho
            rows_A.append(row); rows_b.append(p.E_bar - E0)
            row = np.zeros(n); row[iF(k)] = p.rho
            rows_A.append(row); rows_b.append(E0)
        else:
            row = np.zeros(n); row[iE(k - 1)] =  1.0; row[iF(k)] = p.rho
            rows_A.append(row); rows_b.append(p.E_bar)
            row = np.zeros(n); row[iE(k - 1)] = -1.0; row[iF(k)] = p.rho
            rows_A.append(row); rows_b.append(0.0)

        # (3) Ramp:  |P_k - P_{k-1}| ≤ ΔP̄  (anchor at P_prev for k=0)
        if k == 0:
            row = np.zeros(n); row[iP(k)] =  1.0
            rows_A.append(row); rows_b.append(p.dP_bar + P_prev)
            row = np.zeros(n); row[iP(k)] = -1.0
            rows_A.append(row); rows_b.append(p.dP_bar - P_prev)
        else:
            row = np.zeros(n); row[iP(k)] =  1.0; row[iP(k - 1)] = -1.0
            rows_A.append(row); rows_b.append(p.dP_bar)
            row = np.zeros(n); row[iP(k)] = -1.0; row[iP(k - 1)] =  1.0
            rows_A.append(row); rows_b.append(p.dP_bar)

        # (4) No sell-back:  P + F ≤ L_k
        row = np.zeros(n); row[iP(k)] = 1.0; row[iF(k)] = 1.0
        rows_A.append(row); rows_b.append(Lk)

        # Epigraph of max_k d_k:  D ≥ L_k - P_k + α_k F_k
        #                        -D - P_k + α_k F_k ≤ -L_k
        row = np.zeros(n); row[iD] = -1.0; row[iP(k)] = -1.0; row[iF(k)] = ak
        rows_A.append(row); rows_b.append(-Lk)

    A_ub = np.array(rows_A)
    b_ub = np.array(rows_b)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        return 0.0, 0.0, res.status
    x = res.x
    return float(x[iP(0)]), float(x[iF(0)]), 0


# ─────────────────────────────────────────────────────────────────────────────
# Standalone cost evaluator  (paper eq. IV.11)
# ─────────────────────────────────────────────────────────────────────────────
def cost_on_realization(
    P: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    pi_e: np.ndarray,
    pi_f: np.ndarray,
    alpha: np.ndarray,
    pi_D: float,
) -> Tuple[float, dict]:
    """
    Compute the total monthly cost of applying commitment (P, F) on a given
    realization of (L, pi_e, pi_f, alpha).  Used for Fig-8-style scoring
    where the same commitment is evaluated across many synthetic draws.
    """
    d = L - P + alpha * F
    energy = float(np.sum(pi_e * (alpha * F - P)))
    fr_rev = float(np.sum(pi_f * F))
    demand = float(pi_D * d.max())
    net    = energy - fr_rev + demand
    return net, dict(energy=energy, fr=fr_rev, demand=demand, peak=float(d.max()))

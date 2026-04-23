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

Subclasses implement only `solve_step(t, E0, D_prev, P_prev, F_prev)`.
They are free to use point forecasts (deterministic), scenario draws
(stochastic), or the true future (perfect info) — the base class doesn't
know or care.

The single-path LP builder `solve_single_path_lp` is a module-level helper
used by both DeterministicMPC and PerfectInfoMPC — same LP structure, just
different forecast data fed in.

Delay model
-----------
When `delay=True` is passed to `simulate()`, a one-hour FR lockout delay is
encoded via the state variable A_t with dynamics A_{t+1} = F_t.  The active
FR capacity at time t is A_t = F_{t-1} (with A_0 = 0), and it is A_t (not
F_t) that appears in:
    - SoC dynamics:  E_{t+1} = E_t - P_t + alpha_t * A_t
    - Energy cost:   pi_e_t * (alpha_t * A_t - P_t)
    - FR revenue:    pi_f_t * A_t
    - Grid draw:     d_t = L_t - P_t + alpha_t * A_t  (for peak demand)
    - No-sell-back:  P_t + A_t <= L_t
The LP horizon planner receives A0 = F_{t-1} as a constant and plans F_t
onward; the SoC-buffer feasibility constraints still reference the committed
F[k] (not A[k]) because the battery must be able to honour next period's
commitment.
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
    F: np.ndarray             # (T,)    FR capacity committed [kW]
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

    # ------------------------------------------------------------------
    # Subclass must implement this.
    # F_prev is the FR capacity committed at the previous step (= A_t in
    # delay mode).  It is None when called in no-delay mode so that
    # subclasses written before delay support was added keep working.
    # ------------------------------------------------------------------
    @abstractmethod
    def solve_step(
        self,
        t: int,
        E0: float,
        D_prev: float,
        P_prev: float,
        F_prev: float | None = None,
    ) -> StepResult:
        """
        Return the first-stage (P_{t+1}, F_{t+1}) commitment.

        Parameters
        ----------
        t       : current time index
        E0      : state-of-charge entering this step [kWh]
        D_prev  : running peak demand so far in this billing period [kW]
        P_prev  : net battery power committed at t-1 [kW]
        F_prev  : FR capacity committed at t-1, i.e. A_t [kW].
                  None signals no-delay mode (original behaviour).
        """

    # ------------------------------------------------------------------
    def simulate(
        self,
        L: np.ndarray,
        pi_e: np.ndarray,
        pi_f: np.ndarray,
        alpha: np.ndarray,
        verbose: bool = False,
        delay: bool = False,
    ) -> SimResult:
        """
        Receding-horizon simulation over T hours against REAL realizations.

        Parameters
        ----------
        L, pi_e, pi_f, alpha : real realized signal arrays, length T
        verbose : print progress every 24 steps
        delay   : if True, model a one-hour FR lockout delay via A_t = F_{t-1}.
                  The active FR capacity at step t is A_t, not F_t.
                  F_t is the commitment made *this* step and takes effect at t+1.
        """
        T = self.T
        E = np.zeros(T + 1); E[0] = self.E0
        P = np.zeros(T); F = np.zeros(T); d = np.zeros(T)
        cost_e = np.zeros(T); rev_fr = np.zeros(T)
        demand_charges: list[float] = []

        D_run  = 0.0
        P_prev = 0.0
        F_prev = 0.0   # A_0 = 0: no FR was active before the simulation start

        n_status_bad = 0
        t0 = time.time()

        for t in range(T):
            # ── billing period boundary ───────────────────────────────────
            if t > 0 and t % self.M == 0:
                demand_charges.append(self.pi_D * D_run)
                if verbose:
                    print(f"    [{self.name}] billing end t={t}  "
                          f"demand=${self.pi_D * D_run:,.0f}  peak={D_run:.0f} kW")
                D_run = 0.0

            # ── solve LP / policy for this step ──────────────────────────
            # Pass F_prev only in delay mode so that subclasses can use it
            # as A0 in the LP; None recovers original (no-delay) behaviour.
            step = self.solve_step(
                t, E[t], D_run, P_prev,
                F_prev=F_prev if delay else None,
            )

            if step.status != 0:
                n_status_bad += 1
                P_t, F_t = 0.0, 0.0
            else:
                P_t, F_t = step.P, step.F

            # ── active FR capacity for physics this step ──────────────────
            # delay=True  → A_t = F_{t-1}  (committed last step, locked in)
            # delay=False → A_t = F_t      (original: immediate effect)
            A_t = F_prev if delay else F_t

            # ── apply true realized signals ───────────────────────────────
            a_t = alpha[t]
            L_t = L[t]

            E[t + 1]  = float(np.clip(E[t] - P_t + a_t * A_t, 0.0, self.p.E_bar))
            d_t       = L_t - P_t + a_t * A_t

            P[t]      = P_t
            F[t]      = F_t           # store the commitment, not A_t
            d[t]      = d_t
            cost_e[t] = pi_e[t] * (a_t * A_t - P_t)
            rev_fr[t] = pi_f[t] * A_t   # revenue on *delivered* (active) capacity
            D_run     = max(D_run, d_t)

            # ── advance state ─────────────────────────────────────────────
            P_prev = P_t
            F_prev = F_t   # A_{t+1} = F_t

            if verbose and t % 24 == 0:
                print(
                    f"    [{self.name}] t={t:4d}  SoC={E[t]:6.0f}  "
                    f"P={P_t:+7.1f}  F={F_t:6.1f}  A={A_t:6.1f}  peak={D_run:7.0f}"
                )

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
    A0: float | None = None,
) -> Tuple[float, float, int]:
    """
    Solve the deterministic / perfect-info horizon LP.

    The only difference between deterministic and perfect-info MPC is what
    goes into (L, pi_e, pi_f, alpha) — mean forecasts vs. true future.  The
    LP structure is identical in both cases.

    Delay model
    -----------
    A0 is the active FR capacity entering the horizon (= F committed at t-1).
    When A0 is None the original no-delay formulation is used: A[k] = F[k].
    When A0 is a float the delay formulation is used:
        A[0]   = A0          (constant — already committed, not a decision var)
        A[k]   = F[k-1]      for k >= 1  (LP variable from previous horizon step)

    In both modes the LP variable layout is unchanged:
        x = [P_0..P_{h-1}, F_0..F_{h-1}, E_0..E_{h-1}, D]
    A[k] only replaces F[k] in the physics (dynamics, cost, demand epigraph,
    no-sell-back).  The SoC-buffer feasibility constraints keep F[k] because
    the battery must be able to honour the *current* commitment next period.

    Parameters
    ----------
    horizon            : planning horizon h
    E0                 : state-of-charge at start of horizon [kWh]
    D_prev             : running peak at start of horizon [kW]
    P_prev             : P committed at t-1 (ramp anchor) [kW]
    L, pi_e, pi_f, alpha : forecast arrays of length >= h
    params             : SystemParams
    pi_D               : demand charge rate [$/kW]
    sigma              : number of billing periods (scales demand term)
    gamma              : discount factor (default 1.0)
    A0                 : active FR capacity entering horizon (delay mode).
                         None → no-delay (original behaviour).

    Returns
    -------
    (P_0, F_0, status) — status 0 means optimal.
    """
    h = horizon
    p = params
    n = 3 * h + 1

    # ── index helpers ────────────────────────────────────────────────────────
    def iP(k): return k
    def iF(k): return h + k
    def iE(k): return 2 * h + k
    iD = 3 * h

    use_delay = A0 is not None

    def A_coeff(k):
        """
        Returns (lp_col, const_val) for A[k].

        If use_delay:
            k == 0  → A[0] is the constant A0; lp_col=None, const_val=A0
            k >= 1  → A[k] = F[k-1];           lp_col=iF(k-1), const_val=0
        Else:
            A[k] = F[k];                        lp_col=iF(k), const_val=0
        """
        if not use_delay:
            return iF(k), 0.0
        if k == 0:
            return None, float(A0)
        return iF(k - 1), 0.0

    # ── Objective ────────────────────────────────────────────────────────────
    # min  Σ_k γ^k [ π^e_k (α_k A_k - P_k) - π^f_k A_k ]  +  (π^D / σ) D
    #
    # Constant A0 terms (k=0 in delay mode) do not affect the optimisation
    # and are dropped from c[].
    c = np.zeros(n)
    gpow = gamma ** np.arange(h)

    for k in range(h):
        Acol, _ = A_coeff(k)
        c[iP(k)] = -gpow[k] * pi_e[k]
        if Acol is not None:
            c[Acol] += gpow[k] * (pi_e[k] * alpha[k] - pi_f[k])
    c[iD] = pi_D / sigma

    # ── Variable bounds ───────────────────────────────────────────────────────
    bounds = (
        [(-p.P_und, p.P_bar)] * h +   # P_k
        [(0.0,      p.P_bar)] * h +   # F_k  (committed capacity, always >= 0)
        [(0.0,      p.E_bar)] * h +   # E_k
        [(D_prev,   None)]            # D    (epigraph var, lb = current peak)
    )

    # ── Equality constraints: SoC dynamics ───────────────────────────────────
    # E_k - E_{k-1} + P_k - α_k A_k = 0   (E_{-1} := E0)
    # Rearranged with E0 on the RHS for k=0.
    A_eq = np.zeros((h, n))
    b_eq = np.zeros(h)

    for k in range(h):
        Acol, Aval = A_coeff(k)
        A_eq[k, iE(k)] =  1.0
        A_eq[k, iP(k)] =  1.0
        if Acol is not None:
            A_eq[k, Acol] -= alpha[k]      # -α_k F[k-1] (or F[k]) on LHS
        if k == 0:
            b_eq[k] = E0 + alpha[k] * Aval  # constant A0 contribution → RHS
        else:
            A_eq[k, iE(k - 1)] = -1.0
            # Aval == 0 for k >= 1 in delay mode; nothing added to b_eq

    # ── Inequality constraints ────────────────────────────────────────────────
    rows_A: list[np.ndarray] = []
    rows_b: list[float]      = []

    def add(row, rhs):
        rows_A.append(row)
        rows_b.append(rhs)

    for k in range(h):
        Acol, Aval = A_coeff(k)
        ak = alpha[k]
        Lk = L[k]

        # (1a)  P_k + F_k ≤ P_bar   — joint discharge+FR vs inverter limit
        row = np.zeros(n); row[iP(k)] =  1.0; row[iF(k)] = 1.0
        add(row, p.P_bar)

        # (1b)  -P_k + F_k ≤ P_und  — joint charge+FR vs inverter limit
        row = np.zeros(n); row[iP(k)] = -1.0; row[iF(k)] = 1.0
        add(row, p.P_und)

        # (2b)  E_k + ρ F_k ≤ E_bar }  SoC buffer for *committed* F_k
        # (2b)  -E_k + ρ F_k ≤ 0    }  (feasibility of next-step FR delivery)
        row = np.zeros(n); row[iE(k)] =  1.0; row[iF(k)] = p.rho
        add(row, p.E_bar)
        row = np.zeros(n); row[iE(k)] = -1.0; row[iF(k)] = p.rho
        add(row, 0.0)

        # (2a)  SoC buffer on E_{k-1} (use scalar E0 if k==0)
        if k == 0:
            # ρ F_k ≤ E_bar - E0
            row = np.zeros(n); row[iF(k)] = p.rho
            add(row, p.E_bar - E0)
            # ρ F_k ≤ E0
            row = np.zeros(n); row[iF(k)] = p.rho
            add(row, E0)
        else:
            row = np.zeros(n); row[iE(k - 1)] =  1.0; row[iF(k)] = p.rho
            add(row, p.E_bar)
            row = np.zeros(n); row[iE(k - 1)] = -1.0; row[iF(k)] = p.rho
            add(row, 0.0)

        # (3)  Ramp:  |P_k - P_{k-1}| ≤ ΔP̄
        if k == 0:
            row = np.zeros(n); row[iP(k)] =  1.0
            add(row, p.dP_bar + P_prev)
            row = np.zeros(n); row[iP(k)] = -1.0
            add(row, p.dP_bar - P_prev)
        else:
            row = np.zeros(n); row[iP(k)] =  1.0; row[iP(k - 1)] = -1.0
            add(row, p.dP_bar)
            row = np.zeros(n); row[iP(k)] = -1.0; row[iP(k - 1)] =  1.0
            add(row, p.dP_bar)

        # (4)  No sell-back:  P_k + A_k ≤ L_k
        # (Constant A0 moves to RHS.)
        row = np.zeros(n); row[iP(k)] = 1.0
        if Acol is not None:
            row[Acol] += 1.0
        add(row, Lk - Aval)

        # (5)  Demand epigraph:  D ≥ L_k - P_k + α_k A_k
        #      ↔  -D - P_k + α_k A_k ≤ -L_k
        row = np.zeros(n); row[iD] = -1.0; row[iP(k)] = -1.0
        if Acol is not None:
            row[Acol] += ak
        add(row, -Lk + ak * Aval)

    A_ub = np.array(rows_A)
    b_ub = np.array(rows_b)

    res = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if res.status != 0:
        return 0.0, 0.0, res.status

    return float(res.x[iP(0)]), float(res.x[iF(0)]), 0


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

    Note: this evaluator uses F directly (no delay shift).  If you want to
    score a delay-mode run, shift F by one step before calling this function:
        A = np.concatenate([[0.0], F[:-1]])
        cost_on_realization(P, A, ...)
    """
    d      = L - P + alpha * F
    energy = float(np.sum(pi_e * (alpha * F - P)))
    fr_rev = float(np.sum(pi_f * F))
    demand = float(pi_D * d.max())
    net    = energy - fr_rev + demand
    return net, dict(energy=energy, fr=fr_rev, demand=demand, peak=float(d.max()))
"""
Two-stage stochastic MPC  (paper Sec. III, eq. III.10).

At each hour t, draw |Ξ̄| = S joint scenarios of (L, π^e, π^f, α) over the
horizon and solve:

    min   E_ξ[  Σ_k γ^{k-1} (-π^e_k P_k(ξ) + (π^e_k α_k(ξ) - π^f_k(ξ)) F_k(ξ))
              + (π^D / σ) D(ξ) ]

subject to:
    - per-scenario SoC dynamics, ramp, no-sell-back, SoC buffer, peak-demand
      epigraph,
    - non-anticipativity: P_{s,0} = P_{s',0},  F_{s,0} = F_{s',0} for all s, s'.

The non-anticipativity constraints force a common first-stage commitment
(P*, F*) that every scenario must satisfy.  Later stages are allowed to
diverge per scenario — that's the recourse, which never gets applied but
is what lets the LP price the downstream consequences of today's bid.

Delay model
-----------
When A0 is passed to solve_two_stage_lp (i.e. delay mode), the active FR
capacity at horizon step k is:
    A[0]   = A0          (constant — locked in from last period)
    A[k]   = F[k-1]      for k >= 1
A[k] replaces F[k] in SoC dynamics, the objective, the demand epigraph, and
no-sell-back.  The SoC-buffer feasibility constraints keep F[k] (the current
commitment).  Non-anticipativity still applies to F_{s,0} across scenarios.
A0 is scenario-independent — it is the single real F_{t-1} from the previous
step, not a sampled quantity.

Variable layout  (S scenarios × N horizon, flat vector of length 3·S·N + S):
    iP(s,k) = s·N + k
    iF(s,k) = S·N + s·N + k
    iE(s,k) = 2·S·N + s·N + k
    iD(s)   = 3·S·N + s
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from mpc import MPCController, StepResult, SystemParams
from scenarios import (
    SeasonalLedoitWolfEstimator,
    EmpiricalLogNormalSampler,
    EmpiricalGaussianSampler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage LP builder
# ─────────────────────────────────────────────────────────────────────────────
def solve_two_stage_lp(
    horizon: int,
    n_scen: int,
    E0: float,
    D_prev: float,
    P_prev: float,
    L_sc: np.ndarray,           # (horizon, S)
    pi_e_sc: np.ndarray,        # (horizon, S)
    pi_f_sc: np.ndarray,        # (horizon, S)
    alpha_sc: np.ndarray,       # (horizon, S)
    params: SystemParams,
    pi_D: float,
    sigma: float,
    gamma: float = 1.0,
    A0: float | None = None,    # ← active FR entering horizon; None = no delay
) -> Tuple[float, float, int]:
    """
    Solve the two-stage stochastic LP.  Returns (P*, F*, status) — the common
    first-stage commitment and the LP solver status (0 == optimal).

    A0 controls the delay model (see module docstring).  None recovers the
    original no-delay formulation exactly.
    """
    h = horizon
    S = n_scen
    SN = S * h
    nvar = 3 * SN + S

    def iP(s, k): return s * h + k
    def iF(s, k): return SN + s * h + k
    def iE(s, k): return 2 * SN + s * h + k
    def iD(s):    return 3 * SN + s

    p = params
    use_delay = A0 is not None

    def A_coeff(s, k):
        """
        Returns (lp_col, const_val) for A[s, k].
            no delay   → A[s,k] = F[s,k]         lp_col=iF(s,k),   const=0
            delay k==0 → A[s,0] = A0 (constant)  lp_col=None,      const=A0
            delay k>=1 → A[s,k] = F[s,k-1]       lp_col=iF(s,k-1), const=0
        A0 is scenario-independent (same locked-in value for all s).
        """
        if not use_delay:
            return iF(s, k), 0.0
        if k == 0:
            return None, float(A0)
        return iF(s, k - 1), 0.0

    # ── Objective ────────────────────────────────────────────────────────────
    # c[iP(s,k)] =  (γ^k / S) · (-π^e_{s,k})
    # c[iA(s,k)] += (γ^k / S) · ( π^e_{s,k} α_{s,k} - π^f_{s,k})
    # c[iD(s)]   =  (1 / S) · π^D / σ
    # Constant A0 terms (k=0 in delay mode) don't affect optimisation; dropped.
    c = np.zeros(nvar)
    gpow = gamma ** np.arange(h)
    w = gpow / S
    for s in range(S):
        for k in range(h):
            Acol, _ = A_coeff(s, k)
            c[iP(s, k)] = -w[k] * pi_e_sc[k, s]
            if Acol is not None:
                c[Acol] += w[k] * (pi_e_sc[k, s] * alpha_sc[k, s] - pi_f_sc[k, s])
        c[iD(s)] = (1.0 / S) * (pi_D / sigma)

    # ── Equality constraints (sparse COO) ────────────────────────────────────
    Arow, Acol_eq, Aval, beq = [], [], [], []
    row = 0

    # SoC dynamics per (s, k):
    #   E_{s,k} - E_{s,k-1} + P_{s,k} - α_{s,k} A_{s,k} = (E0 if k==0 else 0)
    # All three lists are appended together to keep lengths in sync.
    for s in range(S):
        for k in range(h):
            Acol_lp, Aval_const = A_coeff(s, k)

            # E_{s,k} coefficient
            Arow.append(row); Acol_eq.append(iE(s, k)); Aval.append(1.0)
            # P_{s,k} coefficient
            Arow.append(row); Acol_eq.append(iP(s, k)); Aval.append(1.0)
            # -α A_{s,k} coefficient (LP variable part only; constant → RHS)
            if Acol_lp is not None:
                Arow.append(row); Acol_eq.append(Acol_lp); Aval.append(-alpha_sc[k, s])

            if k == 0:
                # constant A0 contribution moves to RHS
                beq.append(E0 + alpha_sc[k, s] * Aval_const)
            else:
                Arow.append(row); Acol_eq.append(iE(s, k - 1)); Aval.append(-1.0)
                beq.append(0.0)

            row += 1

    # Non-anticipativity: P_{s,0} = P_{0,0}, F_{s,0} = F_{0,0} for s >= 1
    # (Applies to the committed F, not A, which is constant in delay mode.)
    for s in range(1, S):
        Arow += [row, row]; Acol_eq += [iP(s, 0), iP(0, 0)]; Aval += [1.0, -1.0]
        beq.append(0.0); row += 1
        Arow += [row, row]; Acol_eq += [iF(s, 0), iF(0, 0)]; Aval += [1.0, -1.0]
        beq.append(0.0); row += 1

    A_eq = sp.coo_matrix((Aval, (Arow, Acol_eq)), shape=(row, nvar)).tocsr()
    b_eq = np.array(beq)

    # ── Inequality constraints (sparse COO) ──────────────────────────────────
    Urow, Ucol, Uval, bub = [], [], [], []
    row = 0

    def add_row(entries, rhs):
        nonlocal row
        for col, val in entries:
            Urow.append(row); Ucol.append(col); Uval.append(val)
        bub.append(rhs)
        row += 1

    for s in range(S):
        for k in range(h):
            Lk = L_sc[k, s]
            ak = alpha_sc[k, s]
            Acol_lp, Aval_const = A_coeff(s, k)

            # (1a) P + F ≤ P_bar      (1b) -P + F ≤ P_und
            # SoC-buffer feasibility keeps F[k] (committed), not A[k].
            add_row([(iP(s, k),  1.0), (iF(s, k), 1.0)], p.P_bar)
            add_row([(iP(s, k), -1.0), (iF(s, k), 1.0)], p.P_und)

            # (2b) SoC buffer on E_k — committed F[k]
            add_row([(iE(s, k),  1.0), (iF(s, k), p.rho)], p.E_bar)
            add_row([(iE(s, k), -1.0), (iF(s, k), p.rho)], 0.0)

            # (2a) SoC buffer on E_{k-1} — committed F[k]
            if k == 0:
                add_row([(iF(s, k), p.rho)], p.E_bar - E0)
                add_row([(iF(s, k), p.rho)], E0)
            else:
                add_row([(iE(s, k - 1),  1.0), (iF(s, k), p.rho)], p.E_bar)
                add_row([(iE(s, k - 1), -1.0), (iF(s, k), p.rho)], 0.0)

            # (3) Ramp on P
            if k == 0:
                add_row([(iP(s, k),  1.0)], p.dP_bar + P_prev)
                add_row([(iP(s, k), -1.0)], p.dP_bar - P_prev)
            else:
                add_row([(iP(s, k),  1.0), (iP(s, k - 1), -1.0)], p.dP_bar)
                add_row([(iP(s, k), -1.0), (iP(s, k - 1),  1.0)], p.dP_bar)

            # (4) No sell-back:  P + A ≤ L_k  (constant A0 → RHS)
            entries = [(iP(s, k), 1.0)]
            if Acol_lp is not None:
                entries.append((Acol_lp, 1.0))
            add_row(entries, Lk - Aval_const)

            # (5) Demand epigraph:  -D_s - P + α A ≤ -L_k  (constant A0 → RHS)
            entries = [(iD(s), -1.0), (iP(s, k), -1.0)]
            if Acol_lp is not None:
                entries.append((Acol_lp, ak))
            add_row(entries, -Lk + ak * Aval_const)

    A_ub = sp.coo_matrix((Uval, (Urow, Ucol)), shape=(row, nvar)).tocsr()
    b_ub = np.array(bub)

    # ── Variable bounds ───────────────────────────────────────────────────────
    lb = np.empty(nvar); ub = np.empty(nvar)
    lb[0:SN]        = -p.P_und;  ub[0:SN]        = p.P_bar
    lb[SN:2*SN]     = 0.0;       ub[SN:2*SN]     = p.P_bar
    lb[2*SN:3*SN]   = 0.0;       ub[2*SN:3*SN]   = p.E_bar
    lb[3*SN:]       = D_prev;    ub[3*SN:]        = np.inf
    bounds = list(zip(lb, ub))

    # ── Solve ─────────────────────────────────────────────────────────────────
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        return 0.0, 0.0, res.status
    return float(res.x[iP(0, 0)]), float(res.x[iF(0, 0)]), 0


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────
class StochasticMPC(MPCController):
    """Two-stage stochastic RH-MPC with S sampled scenarios per step."""

    name = "stochastic"

    def __init__(
        self,
        lw_load:  SeasonalLedoitWolfEstimator,
        lw_price: SeasonalLedoitWolfEstimator,
        ln_pif:   EmpiricalLogNormalSampler,
        gn_alpha: EmpiricalGaussianSampler,
        n_scen: int = 20,
        base_seed: int = 2024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lw_load  = lw_load
        self.lw_price = lw_price
        self.ln_pif   = ln_pif
        self.gn_alpha = gn_alpha
        self.S = n_scen
        self.base_seed = base_seed

    def _sample_scenarios(self, t: int, horizon: int):
        seed_t = self.base_seed + t
        rng = np.random.default_rng(seed_t)
        s0, s1, s2, s3 = rng.integers(1 << 31, size=4)

        L_sc   = self.lw_load .sample_horizon(t, horizon, self.S, seed=int(s0))
        pie_sc = self.lw_price.sample_horizon(t, horizon, self.S, seed=int(s1))
        pif_sc = self.ln_pif  .sample(shape=(horizon, self.S), seed=int(s2))
        a_sc   = self.gn_alpha.sample(shape=(horizon, self.S), seed=int(s3))

        # Guard the LP from tail-sample extremes
        L_sc   = np.clip(L_sc,   1.0,  None)
        pie_sc = np.clip(pie_sc, 1e-4, None)
        pif_sc = np.clip(pif_sc, 1e-5, None)
        a_sc   = np.clip(a_sc,  -1.0,  1.0)
        return L_sc, pie_sc, pif_sc, a_sc

    def solve_step(
        self,
        t: int,
        E0: float,
        D_prev: float,
        P_prev: float,
        F_prev: float | None = None,   # ← A_t in delay mode; None = no delay
    ) -> StepResult:
        h = min(self.N, self.T - t)
        L_sc, pie_sc, pif_sc, a_sc = self._sample_scenarios(t, h)
        P, F, status = solve_two_stage_lp(
            horizon=h, n_scen=self.S, E0=E0, D_prev=D_prev, P_prev=P_prev,
            L_sc=L_sc, pi_e_sc=pie_sc, pi_f_sc=pif_sc, alpha_sc=a_sc,
            params=self.p, pi_D=self.pi_D, sigma=self.sigma, gamma=self.gamma,
            A0=F_prev,   # ← None → no-delay LP; float → delay LP
        )
        return StepResult(P=P, F=F, status=status)
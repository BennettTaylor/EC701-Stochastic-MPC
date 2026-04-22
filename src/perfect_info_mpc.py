"""
Perfect-information MPC.

At each time step the controller sees the TRUE future values of
(L, π^e, π^f, α) over the next N hours and plans against them.  This is
an omniscient oracle, not implementable in practice — its realized cost is
a lower bound for any causal controller on the same realization, and the
gap to stochastic MPC is the "value of perfect information."

The LP structure is identical to DeterministicMPC; only the forecast data
differ (truth vs. means).
"""

from __future__ import annotations

import numpy as np

from mpc import MPCController, StepResult, solve_single_path_lp


class PerfectInfoMPC(MPCController):
    """MPC with access to the true future realization over the horizon."""

    name = "perfect_info"

    def __init__(
        self,
        L_true: np.ndarray,
        pi_e_true: np.ndarray,
        pi_f_true: np.ndarray,
        alpha_true: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert L_true.size    >= self.T, "L_true must span full simulation"
        assert pi_e_true.size >= self.T
        assert pi_f_true.size >= self.T
        assert alpha_true.size>= self.T
        self.L_true     = L_true
        self.pi_e_true  = pi_e_true
        self.pi_f_true  = pi_f_true
        self.alpha_true = alpha_true

    def solve_step(self, t: int, E0: float, D_prev: float, P_prev: float) -> StepResult:
        h = min(self.N, self.T - t)
        L   = self.L_true    [t : t + h]
        pie = self.pi_e_true [t : t + h]
        pif = self.pi_f_true [t : t + h]
        a   = self.alpha_true[t : t + h]
        P, F, status = solve_single_path_lp(
            horizon=h, E0=E0, D_prev=D_prev, P_prev=P_prev,
            L=L, pi_e=pie, pi_f=pif, alpha=a,
            params=self.p, pi_D=self.pi_D, sigma=self.sigma, gamma=self.gamma,
        )
        return StepResult(P=P, F=F, status=status)

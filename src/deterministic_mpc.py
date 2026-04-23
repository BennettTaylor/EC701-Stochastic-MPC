"""
Deterministic (point-forecast) MPC.

At every time step, replaces each uncertain quantity with its fitted mean
forecast and solves a single-path LP over the horizon.  Same LP structure
as perfect-info MPC — the only difference is what we feed in for
(L, π^e, π^f, α):

                            deterministic              perfect info
    L[t+k]      :  lw_load.S_hourly[t+k]       true realization
    π^e[t+k]    :  lw_price.S_hourly[t+k]      true realization
    π^f[t+k]    :  ln_pif.mean_   (const)      true realization
    α[t+k]      :  gn_alpha.mu    (≈0 const)   true realization

The paper's "deterministic MPC" is exactly this: the MPC problem made
ignorant by replacing random variables with their expectations.  It serves
as a sanity baseline and underperforms stochastic MPC when the objective's
tail risk (the demand-charge term) matters.
"""

from __future__ import annotations

import numpy as np

from mpc import MPCController, StepResult, solve_single_path_lp
from scenarios import (
    SeasonalLedoitWolfEstimator,
    EmpiricalLogNormalSampler,
    EmpiricalGaussianSampler,
)


class DeterministicMPC(MPCController):
    """Point-forecast MPC — uses fitted distribution means as its horizon oracle."""

    name = "deterministic"

    def __init__(
        self,
        lw_load: SeasonalLedoitWolfEstimator,
        lw_price: SeasonalLedoitWolfEstimator,
        ln_pif:   EmpiricalLogNormalSampler,
        gn_alpha: EmpiricalGaussianSampler,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lw_load  = lw_load
        self.lw_price = lw_price
        self.mean_pif   = ln_pif.mean_
        self.mean_alpha = gn_alpha.mu

    def _forecast(self, t: int, horizon: int):
        L_hat   = self.lw_load .mean_forecast(t, horizon)
        pie_hat = self.lw_price.mean_forecast(t, horizon)
        pif_hat = np.full(horizon, self.mean_pif)
        a_hat   = np.full(horizon, self.mean_alpha)
        # Guard against degenerate values in the LP
        L_hat   = np.clip(L_hat,   1.0,  None)
        pie_hat = np.clip(pie_hat, 1e-4, None)
        pif_hat = np.clip(pif_hat, 1e-5, None)
        a_hat   = np.clip(a_hat,   -1.0, 1.0)
        return L_hat, pie_hat, pif_hat, a_hat

    
    def solve_step(
        self,
        t: int,
        E0: float,
        D_prev: float,
        P_prev: float,
        F_prev: float | None = None,   # ← A_t in delay mode; None = no delay
    ) -> StepResult:
        h = min(self.N, self.T - t)
        L, pie, pif, alpha = self._forecast(t, h)
        P, F, status = solve_single_path_lp(
            horizon=h, E0=E0, D_prev=D_prev, P_prev=P_prev,
            L=L, pi_e=pie, pi_f=pif, alpha=alpha,
            params=self.p, pi_D=self.pi_D, sigma=self.sigma, gamma=self.gamma,
            A0=F_prev,   # ← None → no-delay LP; float → delay LP
        )
        return StepResult(P=P, F=F, status=status)

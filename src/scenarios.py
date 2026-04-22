"""
Scenario generators for the uncertain quantities in the paper.

Four fitted distributions, one per uncertain quantity:

    load L       : SeasonalLedoitWolfEstimator   (diurnal baseline + LW residual)
    elec price π^e: SeasonalLedoitWolfEstimator
    FR price π^f : EmpiricalLogNormalSampler
    FR signal α  : EmpiricalGaussianSampler

Each exposes:
    fit(x)                       -> self
    sample(...)                  -> numpy array
    SeasonalLW additionally:  sample_horizon(t, horizon, n_scen)  for MPC.

Used by both deterministic MPC (their means are point forecasts) and
stochastic MPC (their samples are scenarios).
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Scaled Frobenius inner product / norm  (paper convention: ||I_p||^2 = 1)
# ─────────────────────────────────────────────────────────────────────────────
def _sfrob_inner(A: np.ndarray, B: np.ndarray) -> float:
    p = A.shape[0]
    return float(np.einsum("ij,ij->", A, B) / p)


def _sfrob_norm_sq(A: np.ndarray) -> float:
    return _sfrob_inner(A, A)


# ─────────────────────────────────────────────────────────────────────────────
# Reshape a long hourly series into columns of `hours_per_sample` length.
# ─────────────────────────────────────────────────────────────────────────────
def reshape_to_weekly(y: np.ndarray, hours_per_sample: int = 168) -> np.ndarray:
    """(p, n) where p = hours_per_sample, n = floor(T / p)."""
    T = y.size
    n = T // hours_per_sample
    assert n >= 2, f"need >=2 full windows, got T={T}"
    return y[: n * hours_per_sample].reshape(n, hours_per_sample).T


# ─────────────────────────────────────────────────────────────────────────────
# Ledoit–Wolf shrinkage covariance
# ─────────────────────────────────────────────────────────────────────────────
class LedoitWolfEstimator:
    """
    Shrink the sample covariance of mean-centered data toward a scaled identity:

        Σ̃ = (b²/d²) · m · I  +  (a²/d²) · S

    where (m, d², b², a²) are the paper's bias-consistent scalar estimators,
    and (b²/d²) ∈ [0, 1] is the shrinkage intensity.

    Handles the p > n case (e.g. p=168 weekly profile, n=52 weeks of data),
    which is the whole reason we're using LW.
    """

    def __init__(self):
        self.p = None
        self.n = None

    def fit(self, X: np.ndarray) -> "LedoitWolfEstimator":
        assert X.ndim == 2, "X must be (p, n)"
        p, n = X.shape
        self.p, self.n = p, n

        self.mean_ = X.mean(axis=1)
        Xc = X - self.mean_[:, None]
        S = (Xc @ Xc.T) / n
        self.S = S

        I_p = np.eye(p)
        m = _sfrob_inner(S, I_p)
        d2 = _sfrob_norm_sq(S - m * I_p)

        b2_bar = 0.0
        for k in range(n):
            xk = Xc[:, k]
            b2_bar += _sfrob_norm_sq(np.outer(xk, xk) - S)
        b2_bar /= n * n

        b2 = min(b2_bar, d2)
        a2 = d2 - b2

        w_id = b2 / d2 if d2 > 0 else 1.0
        w_s = a2 / d2 if d2 > 0 else 0.0

        self.m, self.d2, self.b2, self.a2 = m, d2, b2, a2
        self.shrinkage_intensity = w_id
        self.Sigma = w_id * m * I_p + w_s * S
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)  # symmetrize
        return self

    def _chol(self) -> np.ndarray:
        jitter = 1e-10 * max(np.trace(self.Sigma) / self.p, 1.0)
        for _ in range(10):
            try:
                return np.linalg.cholesky(self.Sigma + jitter * np.eye(self.p))
            except np.linalg.LinAlgError:
                jitter *= 10
        raise np.linalg.LinAlgError("Sigma not SPD even with jitter")

    def sample(self, n_scenarios: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        L = self._chol()
        Z = rng.standard_normal(size=(self.p, n_scenarios))
        return self.mean_[:, None] + L @ Z


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal baseline + LW on the residual
# ─────────────────────────────────────────────────────────────────────────────
def _smooth_seasonal_baseline(
    y: np.ndarray, window_days: int = 14
) -> np.ndarray:
    """
    Diurnal-mode baseline: for each of the 24 hours of the day, smooth across
    neighbouring `window_days` days. Result has both seasonal level drift
    (summer > winter) AND season-specific diurnal shape.
    """
    T = y.size
    assert T % 24 == 0, f"expected multiple of 24 hours, got {T}"
    days = T // 24
    half = window_days // 2
    kernel = np.ones(2 * half + 1) / (2 * half + 1)

    L2d = y.reshape(days, 24)
    padded = np.pad(L2d, ((half, half), (0, 0)), mode="edge")
    S2d = np.empty_like(L2d)
    for h in range(24):
        S2d[:, h] = np.convolve(padded[:, h], kernel, mode="valid")
    return S2d.reshape(-1)


class SeasonalLedoitWolfEstimator:
    """
    Fit LW to the seasonally-adjusted residual R(t) = y(t) - S(t), where S is
    a 14-day moving-average diurnal baseline.  Samples are generated as
        ŷ(t) = S(t) + R̂(t),  R̂ ~ N(0, Σ̃_LW).

    This matters because the raw weekly series has huge annual drift that
    flat LW would absorb into a single "shift the whole week up" eigen-
    direction, yielding unrealistic scenarios.
    """

    def __init__(self, hours_per_sample: int = 168, window_days: int = 14):
        self.p = hours_per_sample
        self.window_days = window_days
        self.lw = LedoitWolfEstimator()

    def fit(self, y: np.ndarray) -> "SeasonalLedoitWolfEstimator":
        self.S_hourly = _smooth_seasonal_baseline(y, window_days=self.window_days)
        R = y - self.S_hourly
        self.R_weeks = reshape_to_weekly(R, self.p)
        self.lw.fit(self.R_weeks)
        return self

    def sample_horizon(
        self, t_start: int, horizon: int, n_scenarios: int, seed: int | None = None
    ) -> np.ndarray:
        """
        n_scenarios random samples over `horizon` hours starting at index
        `t_start` (wraps the baseline if needed).  Returns (horizon, S).

        The baseline is applied deterministically at each calendar hour,
        the residual is drawn from the LW Gaussian.
        """
        hour_of_week = t_start % self.p
        full_R = self.lw.sample(n_scenarios=n_scenarios, seed=seed)  # (p, S)
        idx_rw = (hour_of_week + np.arange(horizon)) % self.p
        R_scen = full_R[idx_rw, :]

        T = self.S_hourly.size
        idx_bl = (t_start + np.arange(horizon)) % T
        S = self.S_hourly[idx_bl]
        return S[:, None] + R_scen

    def mean_forecast(self, t_start: int, horizon: int) -> np.ndarray:
        """Deterministic point forecast: just the seasonal baseline."""
        T = self.S_hourly.size
        idx = (t_start + np.arange(horizon)) % T
        return self.S_hourly[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Empirical log-normal  — FR capacity price
# ─────────────────────────────────────────────────────────────────────────────
class EmpiricalLogNormalSampler:
    """π^f ~ exp(N(μ_log, σ_log²)).  Strictly positive, heavy-tailed."""

    def __init__(self, clip_quantile: float = 0.999):
        self.clip_quantile = clip_quantile

    def fit(self, x: np.ndarray) -> "EmpiricalLogNormalSampler":
        x = np.asarray(x, dtype=float)
        x = x[x > 0]
        assert x.size > 2, "need at least 3 positive observations"
        logx = np.log(x)
        self.mu_log = float(logx.mean())
        self.sigma_log = float(logx.std(ddof=1))
        self.mean_ = float(np.exp(self.mu_log + 0.5 * self.sigma_log ** 2))
        self._clip_hi = (
            float(np.quantile(x, self.clip_quantile))
            if self.clip_quantile < 1.0 else np.inf
        )
        return self

    def sample(self, shape, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(size=shape)
        return np.minimum(np.exp(self.mu_log + self.sigma_log * z), self._clip_hi)


# ─────────────────────────────────────────────────────────────────────────────
# Empirical Gaussian  — FR dispatch signal α ∈ [-1, 1]
# ─────────────────────────────────────────────────────────────────────────────
class EmpiricalGaussianSampler:
    """α ~ N(μ, σ²) clipped to [-1, 1]."""

    def fit(self, x: np.ndarray) -> "EmpiricalGaussianSampler":
        x = np.asarray(x, dtype=float)
        self.mu = float(x.mean())
        self.sigma = float(x.std(ddof=1))
        return self

    def sample(self, shape, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(size=shape)
        return np.clip(self.mu + self.sigma * z, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: fit all four at once from a Dataset
# ─────────────────────────────────────────────────────────────────────────────
def fit_all(ds, window_days: int = 14):
    """
    Fit the four scenario generators on a full-year Dataset.
    Returns (lw_load, lw_price, ln_pif, gn_alpha).
    """
    lw_load  = SeasonalLedoitWolfEstimator(window_days=window_days).fit(ds.L)
    lw_price = SeasonalLedoitWolfEstimator(window_days=window_days).fit(ds.pi_e)
    ln_pif   = EmpiricalLogNormalSampler().fit(ds.pi_f)
    gn_alpha = EmpiricalGaussianSampler().fit(ds.alpha)
    return lw_load, lw_price, ln_pif, gn_alpha
